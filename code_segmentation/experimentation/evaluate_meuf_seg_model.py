from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
TMD_TOOL = PROJECT_ROOT / "TMD_DiagnosisTool"
SEG_ARCHIVE = PROJECT_ROOT / "code_archive_seg"
for path in [TMD_TOOL, SEG_ARCHIVE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from UNet import Unet  # noqa: E402
from train_tmj_unetpp_cv_precrop import CLASS_NAMES, json_to_mask  # noqa: E402


SEVERITIES = ["mild", "normal", "severe"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
PRED_COLORS = {
    1: (255, 0, 0, 180),
    2: (255, 150, 0, 195),
}
GT_COLORS = {
    "condyle": (0, 255, 255, 255),
    "glenoid_fossa": (190, 0, 255, 255),
}


def apply_clahe(img: np.ndarray, clip_limit: float = 0.01, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit * 255, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def find_image(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def collect_cases(data_root: Path) -> list[dict]:
    rows = []
    for severity in SEVERITIES:
        for json_path in sorted((data_root / severity).glob("*.json")):
            image_path = find_image(json_path)
            if image_path is not None:
                rows.append({"severity": severity, "image": image_path, "json": json_path})
    return rows


def preprocess_image(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    gray_processed = apply_clahe(gray_resized)
    return image, gray_processed


def predict_mask(model: torch.nn.Module, processed_gray: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(processed_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def dice_score(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred_c = pred == class_id
    target_c = target == class_id
    inter = float(np.logical_and(pred_c, target_c).sum())
    denom = float(pred_c.sum() + target_c.sum())
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def normalize_label(label: str) -> str:
    label = label.lower()
    if "condyle" in label or "condoyle" in label:
        return "condyle"
    if "glenoid" in label or "fossa" in label:
        return "glenoid_fossa"
    return label


def draw_gt_labels(image: Image.Image, json_path: Path, target_size: tuple[int, int]) -> Image.Image:
    import json

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    src_w = int(data.get("imageWidth") or target_size[0])
    src_h = int(data.get("imageHeight") or target_size[1])
    sx = target_size[0] / src_w if src_w else 1.0
    sy = target_size[1] / src_h if src_h else 1.0

    layer = Image.new("RGBA", target_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        label = normalize_label(str(shape.get("label", "")))
        color = GT_COLORS.get(label, (255, 255, 255, 255))
        pts = [(float(x) * sx, float(y) * sy) for x, y in points]
        if (shape.get("shape_type") or "polygon") == "polygon" and len(pts) >= 3:
            pts = pts + [pts[0]]
            draw.line(pts, fill=(0, 0, 0, 255), width=9, joint="curve")
            draw.line(pts, fill=color, width=5, joint="curve")
        else:
            draw.line(pts, fill=(0, 0, 0, 255), width=11, joint="curve")
            draw.line(pts, fill=color, width=6, joint="curve")
    return Image.alpha_composite(image, layer)


def add_legend(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    items = [
        ("Pred condyle", PRED_COLORS[1]),
        ("Pred fossa", PRED_COLORS[2]),
        ("GT condyle", GT_COLORS["condyle"]),
        ("GT fossa", GT_COLORS["glenoid_fossa"]),
    ]
    x = y = 8
    width = 150
    line_h = 20
    height = 16 + line_h * len(items)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=(0, 0, 0, 170))
    for idx, (text, color) in enumerate(items):
        yy = y + 8 + idx * line_h
        draw.rectangle((x + 8, yy + 3, x + 22, yy + 17), fill=color)
        draw.text((x + 29, yy + 1), text, fill=(255, 255, 255, 255), font=font)
    return image


def make_overlay(processed_gray: np.ndarray, pred: np.ndarray, json_path: Path) -> Image.Image:
    base = Image.fromarray(processed_gray).convert("RGBA")
    layer = np.zeros((processed_gray.shape[0], processed_gray.shape[1], 4), dtype=np.uint8)
    for class_id, color in PRED_COLORS.items():
        layer[pred == class_id] = color
    result = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))
    result = draw_gt_labels(result, json_path, (processed_gray.shape[1], processed_gray.shape[0]))
    return add_legend(result)


def save_palette_mask(mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    palette = [0, 0, 0, 255, 0, 0, 255, 150, 0]
    palette += [0] * (768 - len(palette))
    image.putpalette(palette)
    image.save(path)


def write_contact_sheets(output_dir: Path) -> None:
    font = ImageFont.load_default()
    for severity_dir in sorted((output_dir / "overlays").iterdir()):
        if not severity_dir.is_dir():
            continue
        files = sorted(severity_dir.glob("*_overlay.png"))[:24]
        if not files:
            continue
        thumb_w, thumb_h = 340, 205
        label_h = 20
        cols = 3
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(files):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (i % cols) * thumb_w + (thumb_w - image.width) // 2
            y = (i // cols) * (thumb_h + label_h)
            sheet.paste(image, (x, y))
            draw.text(((i % cols) * thumb_w + 6, y + thumb_h + 2), path.stem.replace("_overlay", ""), fill=(0, 0, 0), font=font)
        sheet.save(output_dir / f"contact_sheet_{severity_dir.name}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TMD_DiagnosisTool OPG segmentation model on TMJ_clas labels.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMD_DiagnosisTool\model_OPGseg.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\meuf_model_opgseg_eval"))
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--line-thickness", type=int, default=11)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_root = args.output_dir / "masks"
    overlays_root = args.output_dir / "overlays"
    masks_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(channels=[1, 32, 64, 128, 256, 512], no_classes=3, output_size=(args.height, args.width)).float().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals_i = np.zeros(3, dtype=np.float64)
    totals_p = np.zeros(3, dtype=np.float64)
    totals_t = np.zeros(3, dtype=np.float64)
    report_rows = []
    per_image_rows = []

    for case in collect_cases(args.data_root):
        severity = case["severity"]
        image_path = case["image"]
        json_path = case["json"]
        original_image, processed = preprocess_image(image_path, args.width, args.height)
        pred = predict_mask(model, processed, device)
        gt_orig = json_to_mask(json_path, original_image.shape[0], original_image.shape[1], args.line_thickness)
        gt = np.array(Image.fromarray(gt_orig.astype(np.uint8)).resize((args.width, args.height), Image.Resampling.NEAREST))

        mask_dir = masks_root / severity
        overlay_dir = overlays_root / severity
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
        save_palette_mask(pred, mask_path)
        make_overlay(processed, pred, json_path).convert("RGB").save(overlay_path, quality=95)

        image_dice = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            pred_c = pred == class_id
            gt_c = gt == class_id
            inter = float(np.logical_and(pred_c, gt_c).sum())
            pred_count = float(pred_c.sum())
            gt_count = float(gt_c.sum())
            totals_i[class_id] += inter
            totals_p[class_id] += pred_count
            totals_t[class_id] += gt_count
            image_dice[f"dice_{class_name}"] = dice_score(pred, gt, class_id)

        counts = np.bincount(pred.reshape(-1), minlength=3)
        report_rows.append(
            {
                "severity": severity,
                "image": str(image_path),
                "json": str(json_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "background_pixels": int(counts[0]),
                "condyle_pixels": int(counts[1]),
                "glenoid_fossa_pixels": int(counts[2]),
            }
        )
        per_image_rows.append(
            {
                "severity": severity,
                "image": str(image_path),
                "mask": str(mask_path),
                **image_dice,
                "macro_fg_dice": float((image_dice["dice_condyle"] + image_dice["dice_glenoid_fossa"]) / 2.0),
            }
        )

    global_dice = (2.0 * totals_i + 1e-6) / (totals_p + totals_t + 1e-6)
    summary_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        summary_rows.append(
            {
                "class": class_name,
                "dice": float(global_dice[class_id]),
                "target_pixels": int(totals_t[class_id]),
                "pred_pixels": int(totals_p[class_id]),
            }
        )
    summary_rows.append({"class": "macro_foreground", "dice": float(global_dice[1:].mean()), "target_pixels": "", "pred_pixels": ""})
    summary_rows.append({"class": "macro_all", "dice": float(global_dice.mean()), "target_pixels": "", "pred_pixels": ""})

    with (args.output_dir / "prediction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.output_dir / "dice_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (args.output_dir / "dice_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "dice", "target_pixels", "pred_pixels"])
        writer.writeheader()
        writer.writerows(summary_rows)

    write_contact_sheets(args.output_dir)
    print(f"Cases processed: {len(report_rows)}")
    print(f"Output: {args.output_dir}")
    for row in summary_rows:
        print(f"{row['class']}: {row['dice']}")


if __name__ == "__main__":
    main()
