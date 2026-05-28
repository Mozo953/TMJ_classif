from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
SEG_HELPERS = PROJECT_ROOT / "seg_code_useful"
SEG_ARCHIVE = PROJECT_ROOT / "code_archive_seg"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SEG_HELPERS) not in sys.path:
    sys.path.insert(0, str(SEG_HELPERS))
if str(SEG_ARCHIVE) not in sys.path:
    sys.path.insert(0, str(SEG_ARCHIVE))

from infer_cv2_with_fossa_erosion_grid import (  # noqa: E402
    CLASS_NAMES,
    checkpoint_path,
    collect_inference_samples,
    erode_fossa,
    load_model,
    make_contact_sheets,
    make_overlay,
    pred_from_probs,
    predict_probs,
    save_palette_mask,
)


def keep_top_components(
    mask: np.ndarray,
    probs: np.ndarray,
    class_id: int,
    max_components: int,
    rank_by: str,
) -> tuple[np.ndarray, list[dict]]:
    class_mask = (mask == class_id).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
    components = []
    for component_id in range(1, count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        component_probs = probs[class_id][component]
        components.append(
            {
                "component_id": component_id,
                "area_pixels": area,
                "prob_sum": float(component_probs.sum()),
                "prob_mean": float(component_probs.mean()),
                "prob_max": float(component_probs.max()),
                "keep": False,
            }
        )

    rank_key = "area_pixels" if rank_by == "area" else "prob_mean"
    keep_ids = {
        component["component_id"]
        for component in sorted(components, key=lambda item: item[rank_key], reverse=True)[:max_components]
    }
    for component in components:
        component["keep"] = component["component_id"] in keep_ids

    kept_mask = np.isin(labels, list(keep_ids)) if keep_ids else np.zeros_like(class_mask, dtype=bool)
    return kept_mask, components


def filter_mask_top2(mask: np.ndarray, probs: np.ndarray, max_components: int, rank_by: str) -> tuple[np.ndarray, list[dict]]:
    filtered = np.zeros_like(mask, dtype=np.uint8)
    rows = []
    for class_id in [1, 2]:
        kept, components = keep_top_components(mask, probs, class_id, max_components, rank_by)
        filtered[kept] = class_id
        for component in components:
            rows.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], **component})
    return filtered, rows


def read_best_erosion_kernel(source_dir: Path) -> int:
    grid_files = sorted(source_dir.glob("fossa_erosion_grid_tune_fold_*.csv"))
    if not grid_files:
        return 1
    with grid_files[0].open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 1
    best = max(rows, key=lambda row: float(row["dice_fossa"]))
    return int(best["kernel_size"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fold-2 segmentation inference and keep max two condyle/fossa zones."
    )
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_cv02_fossa_erosion_best"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_fold02_fossa_erosion_top2_largest"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\unet_runs\unetpp_cv5_boundary_fossa_threshold"),
    )
    parser.add_argument("--inference-fold", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-x-fraction", type=float, default=0.38)
    parser.add_argument("--crop-y-fraction", type=float, default=0.58)
    parser.add_argument("--fossa-threshold", type=float, default=0.95)
    parser.add_argument("--rank-by", choices=["area", "mean"], default="area")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path(args.run_dir, args.inference_fold), device)
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", args.image_size))
    fossa_threshold = float(checkpoint.get("best_fossa_threshold", args.fossa_threshold))
    x_fraction = float(ckpt_args.get("crop_x_fraction", args.crop_x_fraction))
    y_fraction = float(ckpt_args.get("crop_y_fraction", args.crop_y_fraction))
    erosion_kernel = read_best_erosion_kernel(args.source_dir)

    masks_root = args.output_dir / "masks"
    overlays_root = args.output_dir / "overlays"
    masks_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    component_rows = []
    for sample in collect_inference_samples(args.data_root):
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        probs = predict_probs(model, image, image_size, device, x_fraction, y_fraction)
        raw_mask = pred_from_probs(probs, fossa_threshold, x_fraction, y_fraction)
        raw_mask = erode_fossa(raw_mask, erosion_kernel)
        filtered_mask, components = filter_mask_top2(raw_mask, probs, args.max_components, args.rank_by)

        mask_dir = masks_root / sample.severity
        overlay_dir = overlays_root / sample.severity
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{sample.image_path.stem}_mask.png"
        overlay_path = overlay_dir / f"{sample.image_path.stem}_overlay.png"
        save_palette_mask(filtered_mask, mask_path)
        make_overlay(image, filtered_mask, sample.json_path).convert("RGB").save(overlay_path, quality=95)

        counts = np.bincount(filtered_mask.reshape(-1), minlength=len(CLASS_NAMES))
        report_rows.append(
            {
                "severity": sample.severity,
                "image": str(sample.image_path),
                "json": str(sample.json_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "fossa_threshold": fossa_threshold,
                "fossa_erosion_kernel": erosion_kernel,
                "max_components_per_class": args.max_components,
                "component_rank_by": "area_pixels" if args.rank_by == "area" else "prob_mean",
                "background_pixels": int(counts[0]),
                "condyle_pixels": int(counts[1]),
                "glenoid_fossa_pixels": int(counts[2]),
            }
        )
        for component in components:
            component_rows.append(
                {
                    "severity": sample.severity,
                    "image": str(sample.image_path),
                    "class_id": component["class_id"],
                    "class_name": component["class_name"],
                    "component_id": component["component_id"],
                    "area_pixels": component["area_pixels"],
                    "prob_sum": component["prob_sum"],
                    "prob_mean": component["prob_mean"],
                    "prob_max": component["prob_max"],
                    "keep": component["keep"],
                }
            )

    with (args.output_dir / "prediction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    with (args.output_dir / "component_selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        writer.writerows(component_rows)

    make_contact_sheets(args.output_dir)
    print(f"Checkpoint fold: {checkpoint.get('fold')}")
    print(f"Output: {args.output_dir}")
    print(f"Fossa threshold: {fossa_threshold}")
    print(f"Fossa erosion kernel: {erosion_kernel}x{erosion_kernel}")
    print(f"Max components per class: {args.max_components}")
    print(f"Component rank by: {'area_pixels' if args.rank_by == 'area' else 'prob_mean'}")
    print(f"Masks: {masks_root}")
    print(f"Overlays: {overlays_root}")


if __name__ == "__main__":
    main()
