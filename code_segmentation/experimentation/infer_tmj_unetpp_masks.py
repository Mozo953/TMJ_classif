from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CLASS_NAMES = ["background", "condyle", "glenoid_fossa"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


class Sample:
    def __init__(self, image_path: Path, severity: str):
        self.image_path = image_path
        self.severity = severity


def collect_image_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "unet_runs"]):
        for image_path in sorted(folder.iterdir()):
            if image_path.name.startswith("._"):
                continue
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append(Sample(image_path=image_path, severity=folder.name))
    return samples


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base: int = 32):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = ConvBlock(in_channels, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4])

        self.conv0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.conv1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.conv2_1 = ConvBlock(nb[2] + nb[3], nb[2])
        self.conv3_1 = ConvBlock(nb[3] + nb[4], nb[3])

        self.conv0_2 = ConvBlock(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = ConvBlock(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = ConvBlock(nb[2] * 2 + nb[3], nb[2])

        self.conv0_3 = ConvBlock(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = ConvBlock(nb[1] * 3 + nb[2], nb[1])

        self.conv0_4 = ConvBlock(nb[0] * 4 + nb[1], nb[0])
        self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    @staticmethod
    def upsample(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, size=like.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3, x0_0)], dim=1))
        return self.final(x0_4)


def save_palette_mask(mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    palette = [
        0,
        0,
        0,
        255,
        40,
        40,
        255,
        220,
        0,
    ]
    palette += [0] * (768 - len(palette))
    image.putpalette(palette)
    image.save(path)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colors = np.array([[0, 0, 0], [255, 40, 40], [255, 220, 0]], dtype=np.uint8)
    color_mask = colors[mask]
    foreground = mask > 0
    overlay = image.copy()
    overlay[foreground] = cv2.addWeighted(image, 0.55, color_mask, 0.45, 0)[foreground]
    return overlay


def crop_boxes(width: int, height: int, x_fraction: float, y_fraction: float) -> list[tuple[str, int, int, int, int]]:
    crop_w = int(round(width * x_fraction))
    crop_h = int(round(height * y_fraction))
    crop_w = max(1, min(crop_w, width))
    crop_h = max(1, min(crop_h, height))
    return [
        ("left", 0, 0, crop_w, crop_h),
        ("right", width - crop_w, 0, width, crop_h),
    ]


def predict_resized(model: nn.Module, image: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1)))
    x = x.unsqueeze(0).to(device)
    pred = torch.argmax(model(x), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


def predict_anatomic_crops(
    model: nn.Module,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
    x_fraction: float,
    y_fraction: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    for _, x1, y1, x2, y2 in crop_boxes(width, height, x_fraction, y_fraction):
        crop = image[y1:y2, x1:x2]
        crop_pred = predict_resized(model, crop, image_size, device)
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], crop_pred)
    return full_mask


def apply_anatomic_roi(mask: np.ndarray, x_fraction: float, y_fraction: float) -> np.ndarray:
    height, width = mask.shape[:2]
    roi_mask = np.zeros_like(mask, dtype=bool)
    for _, x1, y1, x2, y2 in crop_boxes(width, height, x_fraction, y_fraction):
        roi_mask[y1:y2, x1:x2] = True
    filtered = mask.copy()
    filtered[~roi_mask] = 0
    return filtered


def apply_anatomic_input_crop(image: np.ndarray, x_fraction: float, y_fraction: float) -> np.ndarray:
    height, width = image.shape[:2]
    cropped = np.zeros_like(image)
    for _, x1, y1, x2, y2 in crop_boxes(width, height, x_fraction, y_fraction):
        cropped[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return cropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TMJ segmentation masks from a trained UNet++ checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\unet_runs\unetpp_cv_5fold_e50\fold_01\best_tmj_unetpp.pt"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_masks_unetpp_cv_fold01_best_valid"),
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--anatomic-crop", action="store_true")
    parser.add_argument("--anatomic-roi-filter", action="store_true")
    parser.add_argument("--preprocess-anatomic-crop", action="store_true")
    parser.add_argument("--crop-x-fraction", type=float, default=0.34)
    parser.add_argument("--crop-y-fraction", type=float, default=0.52)
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    image_size = args.image_size or int(ckpt_args.get("image_size", 512))
    base_channels = int(ckpt_args.get("base_channels", 16))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlus(num_classes=len(CLASS_NAMES), base=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = collect_image_samples(args.data_root)
    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    with torch.no_grad():
        for sample in samples:
            image = np.array(Image.open(sample.image_path).convert("RGB"))
            if args.anatomic_crop:
                pred = predict_anatomic_crops(
                    model,
                    image,
                    image_size,
                    device,
                    args.crop_x_fraction,
                    args.crop_y_fraction,
                )
            else:
                model_input = image
                if args.preprocess_anatomic_crop:
                    model_input = apply_anatomic_input_crop(image, args.crop_x_fraction, args.crop_y_fraction)
                pred = predict_resized(model, model_input, image_size, device)
                if args.anatomic_roi_filter or args.preprocess_anatomic_crop:
                    pred = apply_anatomic_roi(pred, args.crop_x_fraction, args.crop_y_fraction)

            relative_dir = Path(sample.severity)
            out_mask_dir = masks_dir / relative_dir
            out_mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = out_mask_dir / f"{sample.image_path.stem}_mask.png"
            save_palette_mask(pred, mask_path)

            overlay_path = ""
            if args.save_overlays:
                out_overlay_dir = overlays_dir / relative_dir
                out_overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay = make_overlay(image, pred)
                overlay_path = out_overlay_dir / f"{sample.image_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            counts = np.bincount(pred.reshape(-1), minlength=len(CLASS_NAMES))
            report_rows.append(
                {
                    "severity": sample.severity,
                    "image": str(sample.image_path),
                    "mask": str(mask_path),
                    "overlay": str(overlay_path),
                    "anatomic_crop": args.anatomic_crop,
                    "anatomic_roi_filter": args.anatomic_roi_filter,
                    "preprocess_anatomic_crop": args.preprocess_anatomic_crop,
                    "crop_x_fraction": args.crop_x_fraction
                    if args.anatomic_crop or args.anatomic_roi_filter or args.preprocess_anatomic_crop
                    else "",
                    "crop_y_fraction": args.crop_y_fraction
                    if args.anatomic_crop or args.anatomic_roi_filter or args.preprocess_anatomic_crop
                    else "",
                    "background_pixels": int(counts[0]),
                    "condyle_pixels": int(counts[1]),
                    "glenoid_fossa_pixels": int(counts[2]),
                }
            )

    with (args.output_dir / "prediction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "severity",
                "image",
                "mask",
                "overlay",
                "anatomic_crop",
                "anatomic_roi_filter",
                "preprocess_anatomic_crop",
                "crop_x_fraction",
                "crop_y_fraction",
                "background_pixels",
                "condyle_pixels",
                "glenoid_fossa_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images processed: {len(report_rows)}")
    print(f"Anatomic crop inference: {args.anatomic_crop}")
    print(f"Anatomic ROI filter: {args.anatomic_roi_filter}")
    print(f"Preprocess anatomic crop: {args.preprocess_anatomic_crop}")
    if args.anatomic_crop or args.anatomic_roi_filter or args.preprocess_anatomic_crop:
        print(f"Crop fractions: x={args.crop_x_fraction}, y={args.crop_y_fraction}")
    print(f"Masks saved: {masks_dir}")
    if args.save_overlays:
        print(f"Overlays saved: {overlays_dir}")
    print(f"Report saved: {args.output_dir / 'prediction_report.csv'}")


if __name__ == "__main__":
    main()
