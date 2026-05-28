from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch import nn
import torch.nn.functional as F


CLASS_NAMES = ["background", "condyle", "glenoid_fossa"]
CLASS_COLORS = {
    1: (255, 0, 0),       # condyle
    2: (255, 150, 0),     # glenoid fossa
}
GT_LINE_COLORS = {
    "condyle": (0, 220, 255, 255),
    "glenoid_fossa": (0, 255, 120, 255),
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = ConvBlock(in_channels, channels[0])
        self.conv1_0 = ConvBlock(channels[0], channels[1])
        self.conv2_0 = ConvBlock(channels[1], channels[2])
        self.conv3_0 = ConvBlock(channels[2], channels[3])
        self.conv4_0 = ConvBlock(channels[3], channels[4])

        self.conv0_1 = ConvBlock(channels[0] + channels[1], channels[0])
        self.conv1_1 = ConvBlock(channels[1] + channels[2], channels[1])
        self.conv2_1 = ConvBlock(channels[2] + channels[3], channels[2])
        self.conv3_1 = ConvBlock(channels[3] + channels[4], channels[3])

        self.conv0_2 = ConvBlock(channels[0] * 2 + channels[1], channels[0])
        self.conv1_2 = ConvBlock(channels[1] * 2 + channels[2], channels[1])
        self.conv2_2 = ConvBlock(channels[2] * 2 + channels[3], channels[2])

        self.conv0_3 = ConvBlock(channels[0] * 3 + channels[1], channels[0])
        self.conv1_3 = ConvBlock(channels[1] * 3 + channels[2], channels[1])

        self.conv0_4 = ConvBlock(channels[0] * 4 + channels[1], channels[0])
        self.final = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return self.final(x0_4)


def normalize_label(label: str) -> str:
    label = label.lower()
    if "condyle" in label:
        return "condyle"
    if "glenoid" in label or "fossa" in label:
        return "glenoid_fossa"
    return label


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    base_channels = int(args.get("base_channels", 16))
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    model = UNetPlusPlus(num_classes=len(class_names), base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_probabilities(
    model: nn.Module,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    full_probs = []
    for class_idx in range(probs.shape[0]):
        class_prob = cv2.resize(probs[class_idx], original_size, interpolation=cv2.INTER_LINEAR)
        full_probs.append(class_prob)
    return np.stack(full_probs, axis=0).astype(np.float32)


def read_existing_mask(row: dict[str, str], size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(row["mask"])
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask)


def probability_heat_color(prob: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    color = np.zeros((*prob.shape, 4), dtype=np.uint8)
    color[..., 0] = rgb[0]
    color[..., 1] = rgb[1]
    color[..., 2] = rgb[2]
    color[..., 3] = (prob * 235).astype(np.uint8)
    return color


def overlay_probabilities(base: Image.Image, probs: np.ndarray, pred_mask: np.ndarray) -> Image.Image:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    outline_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline_layer)

    for class_idx, rgb in CLASS_COLORS.items():
        class_area = pred_mask == class_idx
        if not np.any(class_area):
            continue
        class_prob = np.where(class_area, probs[class_idx], 0.0)
        heat = Image.fromarray(probability_heat_color(class_prob, rgb), mode="RGBA")
        layer = Image.alpha_composite(layer, heat)

        binary = Image.fromarray(np.where(class_area, 255, 0).astype("uint8"), mode="L")
        edge = binary.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
        outline_draw.bitmap((0, 0), edge, fill=(*rgb, 255))

    return Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), layer), outline_layer)


def overlay_labels(image: Image.Image, json_path: Path) -> Image.Image:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    src_w = int(data.get("imageWidth") or image.size[0])
    src_h = int(data.get("imageHeight") or image.size[1])
    sx = image.size[0] / src_w if src_w else 1.0
    sy = image.size[1] / src_h if src_h else 1.0

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        label = normalize_label(str(shape.get("label", "")))
        color = GT_LINE_COLORS.get(label, (70, 170, 255, 255))
        pts = [(float(x) * sx, float(y) * sy) for x, y in points]
        if (shape.get("shape_type") or "polygon") == "polygon" and len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=color, width=3, joint="curve")
        else:
            draw.line(pts, fill=color, width=5, joint="curve")
    return Image.alpha_composite(image, layer)


def add_legend(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    items = [
        ("Red: condyle prob", (255, 0, 0, 255)),
        ("Orange: fossa prob", (255, 150, 0, 255)),
        ("Cyan: GT condyle", (0, 220, 255, 255)),
        ("Green: GT fossa", (0, 255, 120, 255)),
    ]
    pad = 8
    line_h = 19
    width = 155
    height = pad * 2 + line_h * len(items)
    draw.rounded_rectangle((pad, pad, pad + width, pad + height), radius=6, fill=(0, 0, 0, 155))
    for i, (text, color) in enumerate(items):
        y = pad + pad + i * line_h
        draw.rectangle((pad + 8, y + 2, pad + 22, y + 16), fill=color)
        draw.text((pad + 30, y), text, fill=(255, 255, 255, 255), font=font)
    return image


def save_pixel_csv(csv_path: Path, probs: np.ndarray, pred_mask: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ys, xs = np.where(pred_mask > 0)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "pred_class_id", "pred_class_name", "pred_probability", "condyle_probability", "glenoid_fossa_probability"])
        for y, x in zip(ys, xs):
            class_idx = int(pred_mask[y, x])
            writer.writerow(
                [
                    int(x),
                    int(y),
                    class_idx,
                    CLASS_NAMES[class_idx],
                    float(probs[class_idx, y, x]),
                    float(probs[1, y, x]),
                    float(probs[2, y, x]),
                ]
            )


def make_contact_sheets(output_root: Path) -> None:
    font = ImageFont.load_default()
    for class_dir in sorted([p for p in output_root.iterdir() if p.is_dir() and p.name in {"mild", "normal", "severe"}]):
        files = sorted(class_dir.glob("*_prob_overlay.png"))[:24]
        if not files:
            continue
        thumb_w, thumb_h = 300, 180
        label_h = 20
        cols = 4
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(files):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (i % cols) * thumb_w + (thumb_w - image.width) // 2
            y = (i // cols) * (thumb_h + label_h)
            sheet.paste(image, (x, y))
            draw.text(((i % cols) * thumb_w + 6, y + thumb_h + 2), path.stem.replace("_prob_overlay", ""), fill=(0, 0, 0), font=font)
        sheet.save(output_root / f"contact_sheet_{class_dir.name}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pixel-wise TMJ probabilities and probability overlays.")
    parser.add_argument("--report", type=Path, default=Path("pred_pre_crop_unetpp_fold01/prediction_report.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("unet_runs/unetpp_cv5_precrop_e50/fold_01/best_tmj_unetpp_precrop.pt"))
    parser.add_argument("--output", type=Path, default=Path("pred_pre_crop_unetpp_fold01/probability_overlays"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for a quick test.")
    args = parser.parse_args()

    report_path = args.report.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    image_size = int(checkpoint.get("args", {}).get("image_size", 512))

    with report_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    index_path = output_root / "probability_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "image", "overlay", "prob_npz", "pixel_csv", "status"])
        writer.writeheader()
        for row in rows:
            image_path = Path(row["image"])
            severity = row.get("severity") or image_path.parent.name
            json_path = image_path.with_suffix(".json")
            try:
                base = Image.open(image_path).convert("RGBA")
                probs = predict_probabilities(model, image_path, image_size, device)
                pred_mask = read_existing_mask(row, base.size)

                out_dir = output_root / severity
                prob_dir = output_root / "prob_maps" / severity
                csv_dir = output_root / "pixel_csv" / severity
                out_dir.mkdir(parents=True, exist_ok=True)
                prob_dir.mkdir(parents=True, exist_ok=True)

                overlay = overlay_probabilities(base, probs, pred_mask)
                if json_path.exists():
                    overlay = overlay_labels(overlay, json_path)
                overlay = add_legend(overlay)

                overlay_path = out_dir / f"{image_path.stem}_prob_overlay.png"
                prob_path = prob_dir / f"{image_path.stem}_probs.npz"
                pixel_csv_path = csv_dir / f"{image_path.stem}_pred_pixels.csv"

                overlay.convert("RGB").save(overlay_path, quality=95)
                np.savez_compressed(prob_path, probabilities=probs, class_names=np.array(CLASS_NAMES))
                save_pixel_csv(pixel_csv_path, probs, pred_mask)
                status = "ok"
            except Exception as exc:
                overlay_path = prob_path = pixel_csv_path = Path("")
                status = f"error: {exc}"

            writer.writerow(
                {
                    "severity": severity,
                    "image": str(image_path),
                    "overlay": str(overlay_path),
                    "prob_npz": str(prob_path),
                    "pixel_csv": str(pixel_csv_path),
                    "status": status,
                }
            )

    make_contact_sheets(output_root)
    print(f"Done: {output_root}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
