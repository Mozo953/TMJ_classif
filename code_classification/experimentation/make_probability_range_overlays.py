from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ["background", "condyle", "glenoid_fossa"]

# Discrete probability bands, useful for choosing thresholds visually.
BANDS = [
    (0.30, 0.50, (44, 123, 182, 95), "30-50%"),
    (0.50, 0.70, (171, 217, 233, 115), "50-70%"),
    (0.70, 0.85, (255, 255, 191, 135), "70-85%"),
    (0.85, 0.95, (253, 174, 97, 165), "85-95%"),
    (0.95, 1.01, (215, 25, 28, 205), "95-100%"),
]

CLASS_OUTLINES = {
    1: (255, 0, 0, 255),
    2: (255, 150, 0, 255),
}

GT_LABEL_COLORS = {
    "condyle": (0, 255, 255, 255),        # cyan, distinct from probability bands
    "glenoid_fossa": (190, 0, 255, 255),  # violet, distinct from probability bands
}


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def resize_probs(probs: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if probs.shape[1:] == (size[1], size[0]):
        return probs
    resized = []
    for class_idx in range(probs.shape[0]):
        resized.append(cv2.resize(probs[class_idx], size, interpolation=cv2.INTER_LINEAR))
    return np.stack(resized, axis=0).astype(np.float32)


def resize_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask)


def normalize_label(label: str) -> str:
    label = label.lower()
    if "condyle" in label:
        return "condyle"
    if "glenoid" in label or "fossa" in label:
        return "glenoid_fossa"
    return label


def overlay_gt_labels(image: Image.Image, json_path: Path) -> Image.Image:
    if not json_path.exists():
        return image

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
        color = GT_LABEL_COLORS.get(label, (255, 255, 255, 255))
        pts = [(float(x) * sx, float(y) * sy) for x, y in points]

        if (shape.get("shape_type") or "polygon") == "polygon" and len(pts) >= 3:
            pts_closed = pts + [pts[0]]
            draw.line(pts_closed, fill=(0, 0, 0, 255), width=7, joint="curve")
            draw.line(pts_closed, fill=color, width=4, joint="curve")
        else:
            draw.line(pts, fill=(0, 0, 0, 255), width=9, joint="curve")
            draw.line(pts, fill=color, width=5, joint="curve")

    return Image.alpha_composite(image, layer)


def make_band_layer(prob: np.ndarray, allowed_area: np.ndarray | None = None) -> Image.Image:
    h, w = prob.shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    area = np.ones((h, w), dtype=bool) if allowed_area is None else allowed_area
    for low, high, color, _label in BANDS:
        band = area & (prob >= low) & (prob < high)
        layer[band] = color
    return Image.fromarray(layer, mode="RGBA")


def draw_pred_outline(draw: ImageDraw.ImageDraw, mask: np.ndarray, class_idx: int, color: tuple[int, int, int, int]) -> None:
    binary = (mask == class_idx).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)


def add_legend(image: Image.Image, title: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    title_font = load_font(16)
    font = load_font(14)
    x = 8
    y = 8
    swatch = 18
    line_h = 23
    gt_items = [
        ("GT condyle", GT_LABEL_COLORS["condyle"]),
        ("GT fossa", GT_LABEL_COLORS["glenoid_fossa"]),
    ]
    width = 190
    height = 34 + (len(BANDS) + len(gt_items)) * line_h + 8
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=(0, 0, 0, 170))
    draw.text((x + 8, y + 7), title, fill=(255, 255, 255, 255), font=title_font)
    for i, (_low, _high, color, label) in enumerate(BANDS):
        yy = y + 32 + i * line_h
        draw.rectangle((x + 10, yy + 2, x + 10 + swatch, yy + 2 + swatch), fill=color)
        draw.text((x + 36, yy + 2), label, fill=(255, 255, 255, 255), font=font)
    for j, (label, color) in enumerate(gt_items):
        yy = y + 40 + (len(BANDS) + j) * line_h
        draw.line((x + 10, yy + 11, x + 10 + swatch, yy + 11), fill=(0, 0, 0, 255), width=6)
        draw.line((x + 10, yy + 11, x + 10 + swatch, yy + 11), fill=color, width=3)
        draw.text((x + 36, yy + 2), label, fill=(255, 255, 255, 255), font=font)
    return image


def render_combined(
    base: Image.Image,
    probs: np.ndarray,
    pred_mask: np.ndarray,
    json_path: Path,
    output_path: Path,
    predicted_only: bool,
) -> None:
    result = base.convert("RGBA")
    for class_idx in [1, 2]:
        allowed_area = (pred_mask == class_idx) if predicted_only else None
        layer = make_band_layer(probs[class_idx], allowed_area)
        result = Image.alpha_composite(result, layer)

    draw = ImageDraw.Draw(result)
    for class_idx, color in CLASS_OUTLINES.items():
        draw_pred_outline(draw, pred_mask, class_idx, color)

    result = overlay_gt_labels(result, json_path)
    mode = "pred pixels" if predicted_only else "all pixels"
    result = add_legend(result, f"Prob bands ({mode})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path, quality=95)


def render_single_class(
    base: Image.Image,
    prob: np.ndarray,
    pred_mask: np.ndarray,
    class_idx: int,
    json_path: Path,
    output_path: Path,
    predicted_only: bool,
) -> None:
    allowed_area = (pred_mask == class_idx) if predicted_only else None
    result = Image.alpha_composite(base.convert("RGBA"), make_band_layer(prob, allowed_area))
    draw = ImageDraw.Draw(result)
    draw_pred_outline(draw, pred_mask, class_idx, CLASS_OUTLINES[class_idx])
    result = overlay_gt_labels(result, json_path)
    mode = "pred pixels" if predicted_only else "all pixels"
    result = add_legend(result, f"{CLASS_NAMES[class_idx]} ({mode})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path, quality=95)


def make_contact_sheets(output_root: Path) -> None:
    font = ImageFont.load_default()
    for folder_name in ["combined_predicted_only", "condyle_all_pixels", "glenoid_fossa_all_pixels"]:
        folder = output_root / folder_name
        for severity_dir in sorted([p for p in folder.iterdir() if p.is_dir()] if folder.exists() else []):
            files = sorted(severity_dir.glob("*.png"))[:24]
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
                draw.text(((i % cols) * thumb_w + 6, y + thumb_h + 2), path.stem, fill=(0, 0, 0), font=font)
            sheet.save(output_root / f"contact_sheet_{folder_name}_{severity_dir.name}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create probability range overlays for visual threshold selection.")
    parser.add_argument("--report", type=Path, default=Path("pred_pre_crop_unetpp_fold01/prediction_report.csv"))
    parser.add_argument("--prob-root", type=Path, default=Path("pred_pre_crop_unetpp_fold01/probability_overlays/prob_maps"))
    parser.add_argument("--output", type=Path, default=Path("pred_pre_crop_unetpp_fold01/probability_range_overlays"))
    args = parser.parse_args()

    report_path = args.report.resolve()
    prob_root = args.prob_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with report_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    index_path = output_root / "probability_range_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as index_file:
        fieldnames = [
            "severity",
            "image",
            "combined_predicted_only",
            "combined_all_pixels",
            "condyle_all_pixels",
            "glenoid_fossa_all_pixels",
            "status",
        ]
        writer = csv.DictWriter(index_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_path = Path(row["image"])
            mask_path = Path(row["mask"])
            severity = row.get("severity") or image_path.parent.name
            prob_path = prob_root / severity / f"{image_path.stem}_probs.npz"
            json_path = image_path.with_suffix(".json")
            try:
                base = Image.open(image_path).convert("RGBA")
                probs = resize_probs(np.load(prob_path)["probabilities"], base.size)
                pred_mask = resize_mask(mask_path, base.size)

                combined_pred = output_root / "combined_predicted_only" / severity / f"{image_path.stem}_prob_ranges_pred.png"
                combined_all = output_root / "combined_all_pixels" / severity / f"{image_path.stem}_prob_ranges_all.png"
                condyle_all = output_root / "condyle_all_pixels" / severity / f"{image_path.stem}_condyle_prob_ranges.png"
                fossa_all = output_root / "glenoid_fossa_all_pixels" / severity / f"{image_path.stem}_fossa_prob_ranges.png"

                render_combined(base, probs, pred_mask, json_path, combined_pred, predicted_only=True)
                render_combined(base, probs, pred_mask, json_path, combined_all, predicted_only=False)
                render_single_class(base, probs[1], pred_mask, 1, json_path, condyle_all, predicted_only=False)
                render_single_class(base, probs[2], pred_mask, 2, json_path, fossa_all, predicted_only=False)
                status = "ok"
            except Exception as exc:
                combined_pred = combined_all = condyle_all = fossa_all = Path("")
                status = f"error: {exc}"

            writer.writerow(
                {
                    "severity": severity,
                    "image": str(image_path),
                    "combined_predicted_only": str(combined_pred),
                    "combined_all_pixels": str(combined_all),
                    "condyle_all_pixels": str(condyle_all),
                    "glenoid_fossa_all_pixels": str(fossa_all),
                    "status": status,
                }
            )

    make_contact_sheets(output_root)
    print(f"Done: {output_root}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
