from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PRED_COLORS = {
    1: (255, 0, 0, 220),       # predicted condyle
    2: (255, 150, 0, 225),     # predicted glenoid fossa
}

PRED_OUTLINE_COLORS = {
    1: (255, 0, 0, 255),
    2: (255, 185, 0, 255),
}

GT_FILL_COLORS = {
    "condyle": (0, 190, 255, 55),
    "glenoid_fossa": (0, 255, 120, 45),
}

GT_LINE_COLORS = {
    "condyle": (0, 220, 255, 255),
    "glenoid_fossa": (0, 255, 120, 255),
}


def normalize_label(label: str) -> str:
    label = label.lower()
    if "condyle" in label:
        return "condyle"
    if "glenoid" in label or "fossa" in label:
        return "glenoid_fossa"
    return label


def scaled_points(points: list[list[float]], src_size: tuple[int, int], dst_size: tuple[int, int]) -> list[tuple[float, float]]:
    sx = dst_size[0] / src_size[0] if src_size[0] else 1.0
    sy = dst_size[1] / src_size[1] if src_size[1] else 1.0
    return [(float(x) * sx, float(y) * sy) for x, y in points]


def overlay_prediction(base: Image.Image, mask_path: Path) -> Image.Image:
    mask = Image.open(mask_path)
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)

    mask_data = np.array(mask)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    outline_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline_layer)
    for value, color in PRED_COLORS.items():
        class_mask = (mask_data == value)
        class_alpha = Image.fromarray(np.where(class_mask, color[3], 0).astype("uint8"), mode="L")
        class_layer = Image.new("RGBA", base.size, color)
        layer = Image.alpha_composite(layer, Image.composite(class_layer, Image.new("RGBA", base.size), class_alpha))

        class_binary = Image.fromarray(np.where(class_mask, 255, 0).astype("uint8"), mode="L")
        bbox = class_binary.getbbox()
        if bbox:
            edge = class_binary.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
            outline_draw.bitmap((0, 0), edge, fill=PRED_OUTLINE_COLORS[value])

    combined = Image.alpha_composite(base.convert("RGBA"), layer)
    return Image.alpha_composite(combined, outline_layer)


def overlay_labels(image: Image.Image, json_path: Path) -> Image.Image:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    src_size = (
        int(data.get("imageWidth") or image.size[0]),
        int(data.get("imageHeight") or image.size[1]),
    )
    draw_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    for shape in data.get("shapes", []):
        label = normalize_label(str(shape.get("label", "")))
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        pts = scaled_points(points, src_size, image.size)
        shape_type = shape.get("shape_type") or "polygon"
        line_color = GT_LINE_COLORS.get(label, (70, 170, 255, 255))
        fill_color = GT_FILL_COLORS.get(label, (70, 170, 255, 45))

        if shape_type == "polygon" and len(pts) >= 3:
            draw.polygon(pts, fill=fill_color, outline=line_color)
            draw.line(pts + [pts[0]], fill=line_color, width=3, joint="curve")
        else:
            draw.line(pts, fill=line_color, width=5, joint="curve")

    return Image.alpha_composite(image, draw_layer)


def add_legend(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    items = [
        ("Pred condyle", (255, 60, 60, 220)),
        ("Pred fossa", (255, 170, 40, 230)),
        ("GT condyle", (0, 220, 255, 255)),
        ("GT fossa", (0, 255, 120, 255)),
    ]
    pad = 8
    swatch = 14
    line_h = 19
    width = 132
    height = pad * 2 + line_h * len(items)
    x = pad
    y = pad
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=(0, 0, 0, 150))
    for i, (text, color) in enumerate(items):
        yy = y + pad + i * line_h
        draw.rectangle((x + pad, yy + 2, x + pad + swatch, yy + 2 + swatch), fill=color)
        draw.text((x + pad + swatch + 6, yy), text, fill=(255, 255, 255, 255), font=font)
    return image


def make_overlay(row: dict[str, str], output_root: Path) -> tuple[Path | None, str]:
    image_path = Path(row["image"])
    mask_path = Path(row["mask"])
    json_path = image_path.with_suffix(".json")
    severity = row.get("severity") or image_path.parent.name

    if not image_path.exists():
        return None, f"missing image: {image_path}"
    if not mask_path.exists():
        return None, f"missing mask: {mask_path}"
    if not json_path.exists():
        return None, f"missing json: {json_path}"

    base = Image.open(image_path).convert("RGBA")
    composite = overlay_prediction(base, mask_path)
    composite = overlay_labels(composite, json_path)
    composite = add_legend(composite)

    output_dir = output_root / severity
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_pred_vs_label.png"
    composite.convert("RGB").save(output_path, quality=95)
    return output_path, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay predicted TMJ masks with LabelMe JSON labels.")
    parser.add_argument("--report", type=Path, default=Path("pred_pre_crop_unetpp_fold01/prediction_report.csv"))
    parser.add_argument("--output", type=Path, default=Path("pred_pre_crop_unetpp_fold01/label_pred_overlays"))
    args = parser.parse_args()

    report_path = args.report.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]]
    with report_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    log_path = output_root / "overlay_index.csv"
    made = 0
    issues = 0
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "image", "mask", "json", "overlay", "status"])
        writer.writeheader()
        for row in rows:
            overlay_path, status = make_overlay(row, output_root)
            if overlay_path:
                made += 1
            else:
                issues += 1
            image_path = Path(row["image"])
            writer.writerow(
                {
                    "severity": row.get("severity") or image_path.parent.name,
                    "image": str(image_path),
                    "mask": row.get("mask", ""),
                    "json": str(image_path.with_suffix(".json")),
                    "overlay": str(overlay_path or ""),
                    "status": status,
                }
            )

    print(f"Created {made} overlays in {output_root}")
    print(f"Issues: {issues}")
    print(f"Index: {log_path}")


if __name__ == "__main__":
    main()
