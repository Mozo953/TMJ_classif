from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
SEG_ARCHIVE = PROJECT_ROOT / "code_archive_seg"
if str(SEG_ARCHIVE) not in sys.path:
    sys.path.insert(0, str(SEG_ARCHIVE))

from train_tmj_unetpp_cv_precrop import CLASS_NAMES, json_to_mask  # noqa: E402


def norm_stem(path: Path) -> str:
    text = path.stem.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_json_index(data_root: Path) -> dict[str, tuple[Path, str, Path]]:
    index = {}
    for severity in ["mild", "normal", "severe"]:
        for json_path in (data_root / severity).glob("*.json"):
            image_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = json_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is not None:
                index[norm_stem(json_path)] = (json_path, severity, image_path)
    return index


def convert_mask(raw: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros(raw.shape, dtype=np.uint8)
    for src, dst in mapping.items():
        out[raw == src] = dst
    return out


def dice_from_totals(intersection: np.ndarray, pred_pixels: np.ndarray, target_pixels: np.ndarray) -> np.ndarray:
    return (2.0 * intersection + 1e-6) / (pred_pixels + target_pixels + 1e-6)


def eval_mapping(rows: list[dict], mapping: dict[int, int]) -> dict:
    totals_i = np.zeros(3, dtype=np.float64)
    totals_p = np.zeros(3, dtype=np.float64)
    totals_t = np.zeros(3, dtype=np.float64)
    per_image = []
    for row in rows:
        pred = convert_mask(row["raw_mask"], mapping)
        gt = row["gt"]
        if pred.shape != gt.shape:
            gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((pred.shape[1], pred.shape[0]), Image.Resampling.NEAREST))

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
            image_dice[f"dice_{class_name}"] = (2.0 * inter + 1e-6) / (pred_count + gt_count + 1e-6)

        per_image.append(
            {
                "severity": row["severity"],
                "case": row["case"],
                "mask": str(row["mask_path"]),
                "json": str(row["json_path"]),
                **image_dice,
                "macro_fg_dice": (image_dice["dice_condyle"] + image_dice["dice_glenoid_fossa"]) / 2.0,
            }
        )

    dice = dice_from_totals(totals_i, totals_p, totals_t)
    return {
        "dice": dice,
        "macro_fg": float(dice[1:].mean()),
        "macro_all": float(dice.mean()),
        "pred_pixels": totals_p,
        "target_pixels": totals_t,
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Dice for TMD_DiagnosisTool TMJ_seg masks.")
    parser.add_argument("--tool-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMD_DiagnosisTool\TMJ_seg"))
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMD_DiagnosisTool\TMJ_seg\dice_eval"))
    parser.add_argument("--line-thickness", type=int, default=11)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_index = build_json_index(args.data_root)
    rows = []
    unmatched = []
    for mask_path in sorted((args.tool_root / "masks").glob("*.png")):
        key = norm_stem(mask_path)
        if key not in json_index:
            unmatched.append(str(mask_path))
            continue
        json_path, severity, image_path = json_index[key]
        raw_mask = np.array(Image.open(mask_path))
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[..., 0]
        image = np.array(Image.open(image_path).convert("RGB"))
        gt = json_to_mask(json_path, image.shape[0], image.shape[1], args.line_thickness)
        if gt.shape != raw_mask.shape:
            gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((raw_mask.shape[1], raw_mask.shape[0]), Image.Resampling.NEAREST))
        rows.append(
            {
                "case": key,
                "severity": severity,
                "mask_path": mask_path,
                "json_path": json_path,
                "image_path": image_path,
                "raw_mask": raw_mask,
                "gt": gt,
            }
        )

    candidate_mappings = {
        "1,2=condyle__3,4=fossa": {1: 1, 2: 1, 3: 2, 4: 2},
        "1,3=condyle__2,4=fossa": {1: 1, 3: 1, 2: 2, 4: 2},
        "1,4=condyle__2,3=fossa": {1: 1, 4: 1, 2: 2, 3: 2},
        "1,2=fossa__3,4=condyle": {1: 2, 2: 2, 3: 1, 4: 1},
        "1,3=fossa__2,4=condyle": {1: 2, 3: 2, 2: 1, 4: 1},
        "1,4=fossa__2,3=condyle": {1: 2, 4: 2, 2: 1, 3: 1},
    }

    mapping_results = []
    evaluated = {}
    for name, mapping in candidate_mappings.items():
        result = eval_mapping(rows, mapping)
        evaluated[name] = result
        mapping_results.append(
            {
                "mapping": name,
                "dice_background": float(result["dice"][0]),
                "dice_condyle": float(result["dice"][1]),
                "dice_glenoid_fossa": float(result["dice"][2]),
                "macro_foreground": result["macro_fg"],
                "macro_all": result["macro_all"],
            }
        )

    best_row = max(mapping_results, key=lambda row: row["macro_foreground"])
    best_result = evaluated[best_row["mapping"]]

    with (args.output_dir / "mapping_grid_dice.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping_results[0].keys()))
        writer.writeheader()
        writer.writerows(mapping_results)

    summary_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        summary_rows.append(
            {
                "class": class_name,
                "dice": float(best_result["dice"][class_id]),
                "target_pixels": int(best_result["target_pixels"][class_id]),
                "pred_pixels": int(best_result["pred_pixels"][class_id]),
                "mapping": best_row["mapping"],
            }
        )
    summary_rows.append({"class": "macro_foreground", "dice": best_result["macro_fg"], "target_pixels": "", "pred_pixels": "", "mapping": best_row["mapping"]})
    summary_rows.append({"class": "macro_all", "dice": best_result["macro_all"], "target_pixels": "", "pred_pixels": "", "mapping": best_row["mapping"]})

    with (args.output_dir / "dice_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "dice", "target_pixels", "pred_pixels", "mapping"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.output_dir / "dice_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_result["per_image"][0].keys()))
        writer.writeheader()
        writer.writerows(best_result["per_image"])

    with (args.output_dir / "matched_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "severity", "mask", "json"])
        for row in rows:
            writer.writerow([row["case"], row["severity"], row["mask_path"], row["json_path"]])

    if unmatched:
        with (args.output_dir / "unmatched_masks.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(unmatched))

    print(f"Matched masks: {len(rows)}")
    print(f"Unmatched masks: {len(unmatched)}")
    print(f"Best mapping: {best_row['mapping']}")
    for row in summary_rows:
        print(f"{row['class']}: {row['dice']}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
