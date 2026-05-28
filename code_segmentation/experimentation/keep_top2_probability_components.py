from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from infer_cv2_with_fossa_erosion_grid import (
    CLASS_NAMES,
    FOSSA_CLASS_ID,
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


def keep_top_components(mask: np.ndarray, probs: np.ndarray, class_id: int, max_components: int) -> tuple[np.ndarray, list[dict]]:
    class_mask = (mask == class_id).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
    components = []
    for component_id in range(1, count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        prob_sum = float(probs[class_id][component].sum())
        prob_mean = float(probs[class_id][component].mean()) if area else 0.0
        components.append(
            {
                "component_id": component_id,
                "area_pixels": area,
                "prob_sum": prob_sum,
                "prob_mean": prob_mean,
                "keep": False,
            }
        )

    keep_ids = {
        component["component_id"]
        for component in sorted(components, key=lambda item: item["prob_sum"], reverse=True)[:max_components]
    }
    for component in components:
        component["keep"] = component["component_id"] in keep_ids

    filtered_class = np.isin(labels, list(keep_ids)) if keep_ids else np.zeros_like(class_mask, dtype=bool)
    return filtered_class, components


def filter_mask_top2(mask: np.ndarray, probs: np.ndarray, max_components: int) -> tuple[np.ndarray, list[dict]]:
    filtered = np.zeros_like(mask, dtype=np.uint8)
    component_rows = []
    for class_id in [1, 2]:
        kept, components = keep_top_components(mask, probs, class_id, max_components)
        filtered[kept] = class_id
        for component in components:
            component_rows.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    **component,
                }
            )
    return filtered, component_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep at most two condyle/fossa components by probability sum.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument("--source-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_cv02_fossa_erosion_best"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_cv02_fossa_erosion_top2_prob"))
    parser.add_argument("--run-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\unet_runs\unetpp_cv5_boundary_fossa_threshold"))
    parser.add_argument("--inference-fold", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-x-fraction", type=float, default=0.38)
    parser.add_argument("--crop-y-fraction", type=float, default=0.58)
    parser.add_argument("--fossa-threshold", type=float, default=0.95)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path(args.run_dir, args.inference_fold), device)
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", args.image_size))
    fossa_threshold = float(checkpoint.get("best_fossa_threshold", args.fossa_threshold))
    x_fraction = float(ckpt_args.get("crop_x_fraction", args.crop_x_fraction))
    y_fraction = float(ckpt_args.get("crop_y_fraction", args.crop_y_fraction))

    grid_path = args.source_dir / "fossa_erosion_grid_tune_fold_03.csv"
    erosion_kernel = 1
    if grid_path.exists():
        with grid_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = max(rows, key=lambda row: float(row["dice_fossa"]))
            erosion_kernel = int(best["kernel_size"])

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
        filtered_mask, components = filter_mask_top2(raw_mask, probs, args.max_components)

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
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Fossa threshold: {fossa_threshold}")
    print(f"Fossa erosion kernel: {erosion_kernel}x{erosion_kernel}")
    print(f"Max components per class: {args.max_components}")
    print(f"Masks: {masks_root}")
    print(f"Overlays: {overlays_root}")
    print(f"Component selection: {args.output_dir / 'component_selection.csv'}")


if __name__ == "__main__":
    main()
