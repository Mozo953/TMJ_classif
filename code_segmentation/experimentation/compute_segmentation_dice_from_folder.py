from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
SEG_ARCHIVE = PROJECT_ROOT / "code_archive_seg"
if str(SEG_ARCHIVE) not in sys.path:
    sys.path.insert(0, str(SEG_ARCHIVE))

from train_tmj_unetpp_cv_precrop import CLASS_NAMES, apply_pre_crop, json_to_mask  # noqa: E402


def dice_score(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred_c = pred == class_id
    target_c = target == class_id
    inter = float(np.logical_and(pred_c, target_c).sum())
    denom = float(pred_c.sum() + target_c.sum())
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Dice scores for a predicted mask folder against LabelMe JSON labels.")
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_fold02_fossa_erosion_top2_meanprob"),
    )
    parser.add_argument("--line-thickness", type=int, default=11)
    parser.add_argument("--crop-x-fraction", type=float, default=0.38)
    parser.add_argument("--crop-y-fraction", type=float, default=0.58)
    args = parser.parse_args()

    report_path = args.prediction_dir / "prediction_report.csv"
    if not report_path.exists():
        raise RuntimeError(f"Missing prediction report: {report_path}")

    totals = {
        "intersection": np.zeros(len(CLASS_NAMES), dtype=np.float64),
        "pred_pixels": np.zeros(len(CLASS_NAMES), dtype=np.float64),
        "target_pixels": np.zeros(len(CLASS_NAMES), dtype=np.float64),
    }
    per_image_rows = []

    with report_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = Path(row["image"])
            json_path = Path(row["json"])
            mask_path = Path(row["mask"])

            pred = np.array(Image.open(mask_path))
            if pred.ndim == 3:
                pred = pred[..., 0]
            image = np.array(Image.open(image_path).convert("RGB"))
            gt = json_to_mask(json_path, image.shape[0], image.shape[1], args.line_thickness)
            _, gt = apply_pre_crop(image, gt, args.crop_x_fraction, args.crop_y_fraction)
            if gt.shape != pred.shape:
                gt_image = Image.fromarray(gt.astype(np.uint8))
                gt = np.array(gt_image.resize((pred.shape[1], pred.shape[0]), Image.Resampling.NEAREST))

            image_dice = {}
            for class_id, class_name in enumerate(CLASS_NAMES):
                pred_c = pred == class_id
                target_c = gt == class_id
                totals["intersection"][class_id] += np.logical_and(pred_c, target_c).sum()
                totals["pred_pixels"][class_id] += pred_c.sum()
                totals["target_pixels"][class_id] += target_c.sum()
                image_dice[f"dice_{class_name}"] = dice_score(pred, gt, class_id)

            per_image_rows.append(
                {
                    "severity": row["severity"],
                    "image": str(image_path),
                    "mask": str(mask_path),
                    **image_dice,
                    "macro_fg_dice": float((image_dice["dice_condyle"] + image_dice["dice_glenoid_fossa"]) / 2.0),
                }
            )

    global_dice = (2.0 * totals["intersection"] + 1e-6) / (totals["pred_pixels"] + totals["target_pixels"] + 1e-6)
    summary_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        summary_rows.append(
            {
                "class": class_name,
                "dice": float(global_dice[class_id]),
                "target_pixels": int(totals["target_pixels"][class_id]),
                "pred_pixels": int(totals["pred_pixels"][class_id]),
            }
        )
    summary_rows.append({"class": "macro_foreground", "dice": float(global_dice[1:].mean()), "target_pixels": "", "pred_pixels": ""})
    summary_rows.append({"class": "macro_all", "dice": float(global_dice.mean()), "target_pixels": "", "pred_pixels": ""})

    per_image_path = args.prediction_dir / "dice_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_path = args.prediction_dir / "dice_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "dice", "target_pixels", "pred_pixels"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-image scores: {per_image_path}")
    for row in summary_rows:
        print(f"{row['class']}: {row['dice']}")


if __name__ == "__main__":
    main()
