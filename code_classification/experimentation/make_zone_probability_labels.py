from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ["background", "condyle", "glenoid_fossa"]
CLASS_COLORS = {
    1: (255, 0, 0, 230),
    2: (255, 150, 0, 235),
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


def resize_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask)


def resize_probs(probs: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if probs.shape[1:] == (size[1], size[0]):
        return probs
    resized = []
    for c in range(probs.shape[0]):
        resized.append(cv2.resize(probs[c], size, interpolation=cv2.INTER_LINEAR))
    return np.stack(resized, axis=0).astype(np.float32)


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    color: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> None:
    padding_x = 7
    padding_y = 5
    line_gap = 2
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    box_w = max(widths) + padding_x * 2
    box_h = sum(heights) + line_gap * (len(lines) - 1) + padding_y * 2

    x = int(xy[0] - box_w / 2)
    y = int(xy[1] - box_h / 2)
    x = max(2, min(x, image_size[0] - box_w - 2))
    y = max(2, min(y, image_size[1] - box_h - 2))

    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=5, fill=(0, 0, 0, 185), outline=color, width=2)
    yy = y + padding_y
    for line, h in zip(lines, heights):
        draw.text((x + padding_x, yy), line, fill=(255, 255, 255, 255), font=font)
        yy += h + line_gap


def annotate_zones(
    image_path: Path,
    mask_path: Path,
    prob_path: Path,
    output_path: Path,
    min_area: int,
) -> list[dict[str, str | int | float]]:
    image = Image.open(image_path).convert("RGBA")
    mask = resize_mask(mask_path, image.size)
    probs = resize_probs(np.load(prob_path)["probabilities"], image.size)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for class_idx, color in CLASS_COLORS.items():
        class_area = mask == class_idx
        alpha = np.where(class_area, np.clip(probs[class_idx] * color[3], 0, color[3]), 0).astype(np.uint8)
        layer_arr = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
        layer_arr[..., 0] = color[0]
        layer_arr[..., 1] = color[1]
        layer_arr[..., 2] = color[2]
        layer_arr[..., 3] = alpha
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer_arr, mode="RGBA"))

    result = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(result)
    font = load_font(16)
    rows: list[dict[str, str | int | float]] = []

    for class_idx, color in CLASS_COLORS.items():
        binary = (mask == class_idx).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component = labels == component_id
            component_probs = probs[class_idx][component]
            mean_prob = float(component_probs.mean())
            max_prob = float(component_probs.max())
            p10_prob = float(np.percentile(component_probs, 10))
            cx, cy = centroids[component_id]

            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)

            lines = [
                CLASS_NAMES[class_idx],
                f"mean {mean_prob:.0%}  max {max_prob:.0%}",
            ]
            draw_label_box(draw, (int(cx), int(cy)), lines, font, color, image.size)

            rows.append(
                {
                    "image": str(image_path),
                    "class_id": class_idx,
                    "class_name": CLASS_NAMES[class_idx],
                    "component_id": component_id,
                    "area_pixels": area,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "mean_probability": mean_prob,
                    "max_probability": max_prob,
                    "p10_probability": p10_prob,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path, quality=95)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write mean/max probability labels directly on each predicted zone.")
    parser.add_argument("--report", type=Path, default=Path("pred_pre_crop_unetpp_fold01/prediction_report.csv"))
    parser.add_argument("--prob-root", type=Path, default=Path("pred_pre_crop_unetpp_fold01/probability_overlays/prob_maps"))
    parser.add_argument("--output", type=Path, default=Path("pred_pre_crop_unetpp_fold01/zone_probability_labels"))
    parser.add_argument("--min-area", type=int, default=80)
    args = parser.parse_args()

    report_path = args.report.resolve()
    prob_root = args.prob_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with report_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summary_rows: list[dict[str, str | int | float]] = []
    index_path = output_root / "zone_probability_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as index_file:
        index_writer = csv.DictWriter(index_file, fieldnames=["severity", "image", "mask", "prob_npz", "zone_overlay", "status"])
        index_writer.writeheader()

        for row in rows:
            image_path = Path(row["image"])
            mask_path = Path(row["mask"])
            severity = row.get("severity") or image_path.parent.name
            prob_path = prob_root / severity / f"{image_path.stem}_probs.npz"
            output_path = output_root / severity / f"{image_path.stem}_zone_probs.png"
            try:
                zone_rows = annotate_zones(image_path, mask_path, prob_path, output_path, args.min_area)
                for zone_row in zone_rows:
                    zone_row["severity"] = severity
                    zone_row["zone_overlay"] = str(output_path)
                summary_rows.extend(zone_rows)
                status = "ok"
            except Exception as exc:
                status = f"error: {exc}"

            index_writer.writerow(
                {
                    "severity": severity,
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "prob_npz": str(prob_path),
                    "zone_overlay": str(output_path),
                    "status": status,
                }
            )

    summary_path = output_root / "zone_probability_summary.csv"
    fieldnames = [
        "severity",
        "image",
        "zone_overlay",
        "class_id",
        "class_name",
        "component_id",
        "area_pixels",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "mean_probability",
        "max_probability",
        "p10_probability",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done: {output_root}")
    print(f"Index: {index_path}")
    print(f"Zone summary: {summary_path}")


if __name__ == "__main__":
    main()
