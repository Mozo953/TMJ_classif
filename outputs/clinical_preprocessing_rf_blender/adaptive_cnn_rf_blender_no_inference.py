"""
Adaptive CNN + RF blender for Mild / Normal / Severe.

No CNN inference is run here. The script only reuses precomputed CNN
probabilities and RF probabilities.

Default behavior:
- RF probabilities come from RF CV=5 OOF predictions.
- CNN probabilities come from the existing CNN blender OOF CSV.
- Rows without CNN probabilities fall back to RF only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


LABELS = ["Mild", "Normal", "Severe"]
PROB_COLS = [f"prob_{label}" for label in LABELS]
RF_PROB_COLS = [f"rf_prob_{label}" for label in LABELS]
CNN_PROB_COLS = [f"cnn_prob_{label}" for label in LABELS]


def normalize_label(value) -> str:
    text = str(value).strip().lower()
    if text == "mild":
        return "Mild"
    if text == "normal":
        return "Normal"
    if text == "severe":
        return "Severe"
    return str(value).strip()


def extract_case_id(text) -> int | None:
    if pd.isna(text):
        return None
    nums = re.findall(r"\d+", str(text))
    return int(nums[-1]) if nums else None


def row_normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, None)
    sums = probs.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return probs / sums


def load_rf_probs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Case ID", "y_true", *RF_PROB_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RF probability file missing columns: {sorted(missing)}")
    out = df[["Case ID", "y_true", *RF_PROB_COLS]].copy()
    out["Case ID"] = pd.to_numeric(out["Case ID"], errors="coerce").astype("Int64")
    out["y_true"] = out["y_true"].map(normalize_label)
    return out


def load_cnn_probs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Case ID" not in df.columns:
        if "mask" in df.columns:
            df["Case ID"] = df["mask"].map(extract_case_id)
        elif "image" in df.columns:
            df["Case ID"] = df["image"].map(extract_case_id)
        else:
            raise ValueError("CNN file must contain `Case ID`, `mask`, or `image`.")

    rename = {}
    for label in LABELS:
        lower_col = f"prob_{label.lower()}"
        title_col = f"prob_{label}"
        cnn_col = f"cnn_prob_{label}"
        if lower_col in df.columns:
            rename[lower_col] = cnn_col
        elif title_col in df.columns:
            rename[title_col] = cnn_col
        elif cnn_col in df.columns:
            rename[cnn_col] = cnn_col
        else:
            raise ValueError(f"CNN file missing probability column for {label}.")

    out = df.rename(columns=rename)[["Case ID", *CNN_PROB_COLS]].copy()
    out["Case ID"] = pd.to_numeric(out["Case ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Case ID"]).drop_duplicates("Case ID", keep="first")
    out[CNN_PROB_COLS] = row_normalize(out[CNN_PROB_COLS].values)
    return out


def adaptive_weight(cnn_probs: np.ndarray, rf_probs: np.ndarray, has_cnn: np.ndarray, min_cnn_weight: float, max_cnn_weight: float) -> np.ndarray:
    cnn_conf = np.nanmax(cnn_probs, axis=1)
    rf_conf = np.nanmax(rf_probs, axis=1)
    denom = cnn_conf + rf_conf
    denom[denom == 0] = 1.0
    weight = cnn_conf / denom
    weight = np.clip(weight, min_cnn_weight, max_cnn_weight)
    weight[~has_cnn] = 0.0
    return weight


def blend_predictions(
    rf_probs: np.ndarray,
    cnn_probs: np.ndarray,
    has_cnn: np.ndarray,
    min_cnn_weight: float,
    max_cnn_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    cnn_weight = adaptive_weight(cnn_probs, rf_probs, has_cnn, min_cnn_weight, max_cnn_weight)
    blended_probs = (cnn_weight[:, None] * cnn_probs) + ((1.0 - cnn_weight[:, None]) * rf_probs)
    return row_normalize(blended_probs), cnn_weight


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def tune_weight_bounds(y_true: pd.Series, rf_probs: np.ndarray, cnn_probs: np.ndarray, has_cnn: np.ndarray) -> dict:
    rows = []
    for min_w in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for max_w in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.75]:
            if min_w > max_w:
                continue
            probs, _ = blend_predictions(rf_probs, cnn_probs, has_cnn, min_w, max_w)
            pred = pd.Series([LABELS[i] for i in probs.argmax(axis=1)])
            metrics = evaluate(y_true.reset_index(drop=True), pred)
            rows.append({
                "min_cnn_weight": min_w,
                "max_cnn_weight": max_w,
                **metrics,
            })
    grid = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    best = grid.iloc[0].to_dict()
    return {"best": best, "grid": grid}


def run_blender(
    rf_probs_path: Path,
    cnn_probs_path: Path,
    output_dir: Path,
    min_cnn_weight: float,
    max_cnn_weight: float,
    tune_weights: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rf = load_rf_probs(rf_probs_path)
    cnn = load_cnn_probs(cnn_probs_path)

    merged = rf.merge(cnn, on="Case ID", how="left")
    has_cnn = merged[CNN_PROB_COLS].notna().all(axis=1).values
    rf_probs = row_normalize(merged[RF_PROB_COLS].values)
    cnn_probs = merged[CNN_PROB_COLS].fillna(0.0).values
    cnn_probs = row_normalize(cnn_probs)

    tuning_summary = None
    if tune_weights:
        tuning = tune_weight_bounds(merged["y_true"], rf_probs, cnn_probs, has_cnn)
        tuning["grid"].to_csv(output_dir / "blender_weight_grid.csv", index=False)
        min_cnn_weight = float(tuning["best"]["min_cnn_weight"])
        max_cnn_weight = float(tuning["best"]["max_cnn_weight"])
        tuning_summary = tuning["best"]

    blended_probs, cnn_weight = blend_predictions(rf_probs, cnn_probs, has_cnn, min_cnn_weight, max_cnn_weight)
    pred_idx = blended_probs.argmax(axis=1)

    for idx, label in enumerate(LABELS):
        merged[f"blend_prob_{label}"] = blended_probs[:, idx]
    merged["cnn_available"] = has_cnn
    merged["cnn_weight"] = cnn_weight
    merged["blend_pred"] = [LABELS[i] for i in pred_idx]

    metrics = evaluate(merged["y_true"], merged["blend_pred"])
    labels_present = [label for label in LABELS if label in set(merged["y_true"])]
    cm = pd.DataFrame(
        confusion_matrix(merged["y_true"], merged["blend_pred"], labels=labels_present),
        index=[f"true_{label}" for label in labels_present],
        columns=[f"pred_{label}" for label in labels_present],
    )
    report = classification_report(merged["y_true"], merged["blend_pred"], labels=labels_present, zero_division=0)

    summary = {
        "n_rows": int(len(merged)),
        "n_with_cnn": int(has_cnn.sum()),
        "n_rf_only": int((~has_cnn).sum()),
        "labels": LABELS,
        "min_cnn_weight": min_cnn_weight,
        "max_cnn_weight": max_cnn_weight,
        "tune_weights": tune_weights,
        "tuning_best": tuning_summary,
        "metrics": metrics,
        "rf_probs_path": str(rf_probs_path),
        "cnn_probs_path": str(cnn_probs_path),
    }

    merged.to_csv(output_dir / "blender_predictions.csv", index=False)
    cm.to_csv(output_dir / "blender_confusion_matrix.csv")
    (output_dir / "blender_classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "blender_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    repo_dir = project_dir.parent
    parser = argparse.ArgumentParser(description="Adaptive CNN/RF blender without CNN inference.")
    parser.add_argument(
        "--rf-probs",
        default=str(project_dir / "outputs" / "rf_3class_cv5_optuna" / "rf_cv5_oof_predictions.csv"),
    )
    parser.add_argument(
        "--cnn-probs",
        default=str(repo_dir / "TMJ_clas" / "clas_runs" / "blend_resnets_optuna_cv5_verbose" / "trial_020" / "meta_oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "outputs" / "adaptive_cnn_rf_blender"),
    )
    parser.add_argument("--min-cnn-weight", type=float, default=0.15)
    parser.add_argument("--max-cnn-weight", type=float, default=0.75)
    parser.add_argument("--tune-weights", dest="tune_weights", action="store_true", default=True, help="Tune adaptive CNN weight bounds on available OOF predictions.")
    parser.add_argument("--no-tune-weights", dest="tune_weights", action="store_false", help="Use the provided min/max CNN weights without calibration.")
    args = parser.parse_args()

    summary = run_blender(
        rf_probs_path=Path(args.rf_probs),
        cnn_probs_path=Path(args.cnn_probs),
        output_dir=Path(args.output_dir),
        min_cnn_weight=args.min_cnn_weight,
        max_cnn_weight=args.max_cnn_weight,
        tune_weights=args.tune_weights,
    )
    print("Adaptive blender completed.")
    print(f"Rows: {summary['n_rows']}")
    print(f"Rows with CNN probs: {summary['n_with_cnn']}")
    print(f"RF-only rows: {summary['n_rf_only']}")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value:.4f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
