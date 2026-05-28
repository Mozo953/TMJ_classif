from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = ["background", "condyle", "glenoid_fossa"]
CONDYLE_LABELS = {"R_condyle", "L_condyle", "R_condoyle", "L_condoyle"}
FOSSA_LABELS = {"R_glenoid_fossa", "L_glenoid_fossa", "R_glennoidfossa", "L_glennoidfossa"}
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
DATA_FOLDERS = {"mild", "normal", "severe"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    json_path: Path
    severity: str


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.uint8)
    if target_shape is not None and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def polygon_points(shape: dict) -> np.ndarray:
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.round(points).astype(np.int32)


def json_to_mask(json_path: Path, height: int, width: int, line_thickness: int = 11) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label in CONDYLE_LABELS:
            class_id = 1
        elif label in FOSSA_LABELS:
            class_id = 2
        else:
            continue

        pts = polygon_points(shape)
        if pts.size == 0:
            continue

        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "linestrip" or pts.shape[0] < 3:
            cv2.polylines(mask, [pts], isClosed=False, color=class_id, thickness=line_thickness)
        else:
            cv2.fillPoly(mask, [pts], color=class_id)
    return mask


def collect_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir() and p.name in DATA_FOLDERS]):
        for json_path in sorted(folder.glob("*.json")):
            if json_path.name.startswith("._"):
                continue
            image_path = None
            for ext in IMAGE_EXTS:
                candidate = json_path.with_suffix(ext)
                if candidate.exists() and not candidate.name.startswith("._"):
                    image_path = candidate
                    break
            if image_path is not None:
                samples.append(Sample(image_path=image_path, json_path=json_path, severity=folder.name))
    return samples


def blend_region(base: np.ndarray, region: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    if not np.any(region):
        return base
    color_layer = np.zeros_like(base)
    color_layer[:] = color
    blended = cv2.addWeighted(base, 1.0 - alpha, color_layer, alpha, 0)
    out = base.copy()
    out[region] = blended[region]
    return out


def fill_class_mask(image: np.ndarray, mask: np.ndarray, colors: dict[int, tuple[int, int, int]], alpha: float) -> np.ndarray:
    overlay = image.copy()
    for class_id, color in colors.items():
        overlay = blend_region(overlay, mask == class_id, color, alpha)
    return overlay


def draw_prediction_contours(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    contour_colors = {
        1: (255, 0, 0),
        2: (0, 128, 255),
    }
    for class_id, color in contour_colors.items():
        class_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, color, thickness=3)
    return overlay


def make_prediction_vs_real_overlay(image: np.ndarray, real_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    if real_mask.shape != pred_mask.shape:
        raise ValueError("Le masque réel et le masque prédit doivent avoir la même taille.")

    real_colors = {
        1: (0, 255, 0),
        2: (255, 220, 0),
    }

    overlay = fill_class_mask(image, real_mask, real_colors, 0.35)
    overlay = draw_prediction_contours(overlay, pred_mask)
    return overlay


def resolve_mask_path(mask_dir: Path, stem: str) -> Path | None:
    candidates = [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}_mask.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_image_files(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [path for path in sorted(image_dir.rglob("*")) if path.is_file() and path.suffix.lower() in exts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create overlays that compare prediction masks against real masks.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument(
        "--pred-mask-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_masks_unetpp_cv_fold01_best_valid\masks"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_seg\overlay_pred_vs_real"))
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of images processed. 0 means all images.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.data_root)
    if args.limit > 0:
        samples = samples[: args.limit]

    report_rows: list[tuple[str, str, str, str]] = []
    for sample in samples:
        pred_mask_path = resolve_mask_path(args.pred_mask_dir, sample.image_path.stem)
        if pred_mask_path is None:
            continue

        image = load_rgb_image(sample.image_path)
        real_mask = json_to_mask(sample.json_path, image.shape[0], image.shape[1])
        pred_mask = load_mask(pred_mask_path, image.shape[:2])

        overlay = make_prediction_vs_real_overlay(image, real_mask, pred_mask)
        output_path = args.output_dir / f"{sample.image_path.stem}_overlay.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        report_rows.append((sample.image_path.name, sample.json_path.name, pred_mask_path.name, output_path.name))

    report_path = args.output_dir / "overlay_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "real_json", "pred_mask", "overlay"])
        writer.writerows(report_rows)

    print(f"Saved overlays: {args.output_dir}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()