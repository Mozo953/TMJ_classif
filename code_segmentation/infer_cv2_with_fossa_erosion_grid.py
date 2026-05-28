from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from train_tmj_unetpp_cv_precrop import (
    CLASS_NAMES,
    Sample,
    UNetPlusPlus,
    apply_pre_crop,
    collect_samples,
    crop_boxes,
    json_to_mask,
)


FOSSA_CLASS_ID = 2
EROSION_KERNELS = [1, 3, 5, 7]
PRED_COLORS = {
    1: (255, 0, 0, 190),
    2: (255, 150, 0, 205),
}
GT_COLORS = {
    "condyle": (0, 255, 255, 255),
    "glenoid_fossa": (190, 0, 255, 255),
}


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def checkpoint_path(run_dir: Path, fold: int) -> Path:
    return run_dir / f"fold_{fold:02d}" / "best_tmj_unetpp_boundary_fossa_threshold.pt"


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model = UNetPlusPlus(num_classes=len(CLASS_NAMES), base=int(ckpt_args.get("base_channels", 16))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def read_split_samples(split_path: Path, split_name: str, samples_by_image: dict[str, Sample]) -> list[Sample]:
    selected = []
    with split_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split_name:
                continue
            sample = samples_by_image.get(str(Path(row["image"])))
            if sample is None:
                raise RuntimeError(f"Unknown sample in split: {row['image']}")
            selected.append(sample)
    return selected


def anatomic_roi_mask(shape: tuple[int, int], x_fraction: float, y_fraction: float) -> np.ndarray:
    height, width = shape
    roi = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in crop_boxes(width, height, x_fraction, y_fraction):
        roi[y1:y2, x1:x2] = True
    return roi


def predict_probs(
    model: nn.Module,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
    x_fraction: float,
    y_fraction: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    dummy_mask = np.zeros((height, width), dtype=np.uint8)
    model_input, _ = apply_pre_crop(image, dummy_mask, x_fraction, y_fraction)
    resized = cv2.resize(model_input, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
    full_probs = []
    for class_idx in range(probs.shape[0]):
        full_probs.append(cv2.resize(probs[class_idx], (width, height), interpolation=cv2.INTER_LINEAR))
    return np.stack(full_probs, axis=0).astype(np.float32)


def pred_from_probs(probs: np.ndarray, fossa_threshold: float, x_fraction: float, y_fraction: float) -> np.ndarray:
    pred = np.argmax(probs[:FOSSA_CLASS_ID], axis=0).astype(np.uint8)
    pred[probs[FOSSA_CLASS_ID] > fossa_threshold] = FOSSA_CLASS_ID
    pred[~anatomic_roi_mask(pred.shape, x_fraction, y_fraction)] = 0
    return pred


def erode_fossa(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.copy()
    result = mask.copy()
    fossa = (mask == FOSSA_CLASS_ID).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(fossa, kernel, iterations=1).astype(bool)
    result[mask == FOSSA_CLASS_ID] = 0
    result[eroded] = FOSSA_CLASS_ID
    return result


def dice_for_class(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred_c = pred == class_id
    target_c = target == class_id
    inter = float(np.logical_and(pred_c, target_c).sum())
    denom = float(pred_c.sum() + target_c.sum())
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def tune_erosion_kernel(
    model: nn.Module,
    checkpoint: dict,
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[dict]]:
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", args.image_size))
    fossa_threshold = float(checkpoint.get("best_fossa_threshold", args.fossa_threshold))
    x_fraction = float(ckpt_args.get("crop_x_fraction", args.crop_x_fraction))
    y_fraction = float(ckpt_args.get("crop_y_fraction", args.crop_y_fraction))
    line_thickness = int(ckpt_args.get("line_thickness", args.line_thickness))

    rows = []
    best_kernel = EROSION_KERNELS[0]
    best_dice = -1.0
    for kernel_size in EROSION_KERNELS:
        intersections = 0.0
        pred_pixels = 0.0
        target_pixels = 0.0
        for sample in samples:
            image = np.array(Image.open(sample.image_path).convert("RGB"))
            probs = predict_probs(model, image, image_size, device, x_fraction, y_fraction)
            pred = erode_fossa(pred_from_probs(probs, fossa_threshold, x_fraction, y_fraction), kernel_size)
            gt = json_to_mask(sample.json_path, image.shape[0], image.shape[1], line_thickness)
            _, gt = apply_pre_crop(image, gt, x_fraction, y_fraction)
            pred_fossa = pred == FOSSA_CLASS_ID
            gt_fossa = gt == FOSSA_CLASS_ID
            intersections += float(np.logical_and(pred_fossa, gt_fossa).sum())
            pred_pixels += float(pred_fossa.sum())
            target_pixels += float(gt_fossa.sum())

        dice = (2.0 * intersections + 1e-6) / (pred_pixels + target_pixels + 1e-6)
        rows.append(
            {
                "kernel_size": kernel_size,
                "dice_fossa": dice,
                "pred_fossa_pixels": int(pred_pixels),
                "target_fossa_pixels": int(target_pixels),
            }
        )
        if dice > best_dice:
            best_dice = dice
            best_kernel = kernel_size
    return best_kernel, rows


def save_palette_mask(mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    palette = [0, 0, 0, 255, 0, 0, 255, 150, 0]
    palette += [0] * (768 - len(palette))
    image.putpalette(palette)
    image.save(path)


def normalize_label(label: str) -> str:
    label = label.lower()
    if "condyle" in label or "condoyle" in label:
        return "condyle"
    if "glenoid" in label or "fossa" in label:
        return "glenoid_fossa"
    return label


def draw_gt_labels(image: Image.Image, json_path: Path) -> Image.Image:
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
        color = GT_COLORS.get(label, (255, 255, 255, 255))
        pts = [(float(x) * sx, float(y) * sy) for x, y in points]
        if (shape.get("shape_type") or "polygon") == "polygon" and len(pts) >= 3:
            pts = pts + [pts[0]]
            draw.line(pts, fill=(0, 0, 0, 255), width=11, joint="curve")
            draw.line(pts, fill=color, width=6, joint="curve")
        else:
            draw.line(pts, fill=(0, 0, 0, 255), width=13, joint="curve")
            draw.line(pts, fill=color, width=7, joint="curve")
    return Image.alpha_composite(image, layer)


def make_overlay(image: np.ndarray, mask: np.ndarray, json_path: Path) -> Image.Image:
    base = Image.fromarray(image).convert("RGBA")
    layer = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    for class_id, color in PRED_COLORS.items():
        layer[mask == class_id] = color
    result = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(result)
    for class_id, color in PRED_COLORS.items():
        binary = (mask == class_id).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) < 40:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            draw.rectangle((x, y, x + w, y + h), outline=color[:3] + (255,), width=2)
    result = draw_gt_labels(result, json_path)
    return add_legend(result)


def add_legend(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = load_font(14)
    items = [
        ("Pred condyle", PRED_COLORS[1]),
        ("Pred fossa", PRED_COLORS[2]),
        ("GT condyle", GT_COLORS["condyle"]),
        ("GT fossa", GT_COLORS["glenoid_fossa"]),
    ]
    x = y = 8
    width = 150
    line_h = 22
    height = 16 + line_h * len(items)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=(0, 0, 0, 170))
    for idx, (label, color) in enumerate(items):
        yy = y + 8 + idx * line_h
        draw.rectangle((x + 8, yy + 3, x + 23, yy + 18), fill=color)
        draw.text((x + 30, yy + 2), label, fill=(255, 255, 255, 255), font=font)
    return image


def collect_inference_samples(root: Path) -> list[Sample]:
    return collect_samples(root)


def run_inference(
    model: nn.Module,
    checkpoint: dict,
    samples: list[Sample],
    kernel_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", args.image_size))
    fossa_threshold = float(checkpoint.get("best_fossa_threshold", args.fossa_threshold))
    x_fraction = float(ckpt_args.get("crop_x_fraction", args.crop_x_fraction))
    y_fraction = float(ckpt_args.get("crop_y_fraction", args.crop_y_fraction))

    masks_root = args.output_dir / "masks"
    overlays_root = args.output_dir / "overlays"
    masks_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for sample in samples:
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        probs = predict_probs(model, image, image_size, device, x_fraction, y_fraction)
        pred = pred_from_probs(probs, fossa_threshold, x_fraction, y_fraction)
        pred = erode_fossa(pred, kernel_size)

        mask_dir = masks_root / sample.severity
        overlay_dir = overlays_root / sample.severity
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{sample.image_path.stem}_mask.png"
        overlay_path = overlay_dir / f"{sample.image_path.stem}_overlay.png"
        save_palette_mask(pred, mask_path)
        make_overlay(image, pred, sample.json_path).convert("RGB").save(overlay_path, quality=95)

        counts = np.bincount(pred.reshape(-1), minlength=len(CLASS_NAMES))
        rows.append(
            {
                "severity": sample.severity,
                "image": str(sample.image_path),
                "json": str(sample.json_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "fossa_threshold": fossa_threshold,
                "fossa_erosion_kernel": kernel_size,
                "background_pixels": int(counts[0]),
                "condyle_pixels": int(counts[1]),
                "glenoid_fossa_pixels": int(counts[2]),
            }
        )

    with (args.output_dir / "prediction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(output_dir: Path) -> None:
    font = ImageFont.load_default()
    overlays_root = output_dir / "overlays"
    for severity_dir in sorted([p for p in overlays_root.iterdir() if p.is_dir()] if overlays_root.exists() else []):
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
    parser = argparse.ArgumentParser(description="Tune fossa erosion kernel, then run CV fold inference with masks and GT overlays.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument("--run-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\unet_runs\unetpp_cv5_boundary_fossa_threshold"))
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--inference-fold", type=int, default=2)
    parser.add_argument("--tune-fold", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_cv02_fossa_erosion_best"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--line-thickness", type=int, default=11)
    parser.add_argument("--crop-x-fraction", type=float, default=0.38)
    parser.add_argument("--crop-y-fraction", type=float, default=0.58)
    parser.add_argument("--fossa-threshold", type=float, default=0.95)
    parser.add_argument("--skip-grid", action="store_true", help="Use --erosion-kernel directly instead of tuning on tune fold.")
    parser.add_argument("--erosion-kernel", type=int, default=1, choices=EROSION_KERNELS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.split_dir or args.run_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = collect_samples(args.data_root)
    samples_by_image = {str(sample.image_path): sample for sample in samples}

    if args.skip_grid:
        best_kernel = args.erosion_kernel
        grid_rows = [{"kernel_size": best_kernel, "dice_fossa": "", "pred_fossa_pixels": "", "target_fossa_pixels": ""}]
    else:
        tune_model, tune_checkpoint = load_model(checkpoint_path(args.run_dir, args.tune_fold), device)
        tune_split = split_dir / f"fold_{args.tune_fold:02d}" / "split.csv"
        tune_samples = read_split_samples(tune_split, "valid", samples_by_image)
        best_kernel, grid_rows = tune_erosion_kernel(tune_model, tune_checkpoint, tune_samples, args, device)

    grid_path = args.output_dir / f"fossa_erosion_grid_tune_fold_{args.tune_fold:02d}.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel_size", "dice_fossa", "pred_fossa_pixels", "target_fossa_pixels"])
        writer.writeheader()
        writer.writerows(grid_rows)

    inference_model, inference_checkpoint = load_model(checkpoint_path(args.run_dir, args.inference_fold), device)
    run_inference(inference_model, inference_checkpoint, collect_inference_samples(args.data_root), best_kernel, args, device)
    make_contact_sheets(args.output_dir)

    print(f"Tune fold: {args.tune_fold}")
    print(f"Inference fold: {args.inference_fold}")
    print(f"Best fossa erosion kernel: {best_kernel}x{best_kernel}")
    print(f"Grid saved: {grid_path}")
    print(f"Masks: {args.output_dir / 'masks'}")
    print(f"Overlays: {args.output_dir / 'overlays'}")
    print(f"Report: {args.output_dir / 'prediction_report.csv'}")


if __name__ == "__main__":
    main()
