"""
Retrain the clinical RF and CNN/RF blender on image-matched clinical cases only.

Rules:
- Keep only cases that have BOTH:
  1. clinical row in clinical_merged_with_diag_features.csv
  2. precomputed CNN probabilities in meta_oof_predictions.csv
- Drop clinical cases without CNN probabilities.
- Drop images/CNN rows without clinical case.
- Do not run CNN inference.
- Reuse precomputed CNN OOF probabilities.
- Retrain RF on the matched clinical subset with CV=5, seed=42.
- Train a CV=5 meta-blender between CNN probabilities and RF probabilities.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["Mild", "Normal", "Severe"]
SEED = 42


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
    nums = re.findall(r"\d+", str(text))
    return int(nums[-1]) if nums else None


def row_normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, None)
    sums = probs.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return probs / sums


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def load_cnn(cnn_probs_csv: Path) -> pd.DataFrame:
    cnn = pd.read_csv(cnn_probs_csv)
    if "Case ID" not in cnn.columns:
        if "mask" not in cnn.columns:
            raise ValueError("CNN CSV must contain `Case ID` or `mask`.")
        cnn["Case ID"] = cnn["mask"].map(extract_case_id)
    rename = {
        "prob_mild": "cnn_prob_Mild",
        "prob_normal": "cnn_prob_Normal",
        "prob_severe": "cnn_prob_Severe",
    }
    cnn = cnn.rename(columns=rename)
    required = ["Case ID", "cnn_prob_Mild", "cnn_prob_Normal", "cnn_prob_Severe"]
    missing = [col for col in required if col not in cnn.columns]
    if missing:
        raise ValueError(f"CNN CSV missing columns: {missing}")
    keep = required + [col for col in ["true_label", "pred_label", "mask"] if col in cnn.columns]
    cnn = cnn[keep].copy()
    cnn["Case ID"] = pd.to_numeric(cnn["Case ID"], errors="coerce").astype("Int64")
    cnn = cnn.dropna(subset=["Case ID"]).drop_duplicates("Case ID", keep="first")
    cnn[["cnn_prob_Mild", "cnn_prob_Normal", "cnn_prob_Severe"]] = row_normalize(
        cnn[["cnn_prob_Mild", "cnn_prob_Normal", "cnn_prob_Severe"]].values
    )
    if "true_label" in cnn.columns:
        cnn["cnn_true_label_norm"] = cnn["true_label"].map(normalize_label)
    return cnn


def build_matched_clinical(clinical_csv: Path, cnn_csv: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(clinical_csv)
    clinical["Case ID"] = pd.to_numeric(clinical["Case ID"], errors="coerce").astype("Int64")
    clinical["severity"] = clinical["severity"].map(normalize_label)
    cnn = load_cnn(cnn_csv)

    clinical_ids = set(clinical["Case ID"].dropna().astype(int))
    cnn_ids = set(cnn["Case ID"].dropna().astype(int))
    matched_ids = sorted(clinical_ids & cnn_ids)

    matched = clinical[clinical["Case ID"].isin(matched_ids)].copy()
    matched = matched.sort_values("Case ID").reset_index(drop=True)
    matched_cnn = cnn[cnn["Case ID"].isin(matched_ids)].copy().sort_values("Case ID").reset_index(drop=True)

    dropped = pd.DataFrame({
        "category": (
            ["clinical_without_cnn"] * len(sorted(clinical_ids - cnn_ids))
            + ["cnn_without_clinical"] * len(sorted(cnn_ids - clinical_ids))
        ),
        "Case ID": sorted(clinical_ids - cnn_ids) + sorted(cnn_ids - clinical_ids),
    })
    dropped.to_csv(output_dir / "dropped_unmatched_cases.csv", index=False)

    label_check = matched[["Case ID", "severity"]].merge(
        matched_cnn[["Case ID", "cnn_true_label_norm"]] if "cnn_true_label_norm" in matched_cnn.columns else matched_cnn[["Case ID"]],
        on="Case ID",
        how="left",
    )
    if "cnn_true_label_norm" in label_check.columns:
        label_check["label_match"] = label_check["severity"].eq(label_check["cnn_true_label_norm"])
    label_check.to_csv(output_dir / "label_consistency_check.csv", index=False)

    matched.to_csv(output_dir / "clinical_image_matched_only.csv", index=False)
    matched_cnn.to_csv(output_dir / "cnn_probs_image_matched_only.csv", index=False)
    return matched, matched_cnn


def run_rf(
    matched_csv: Path,
    output_dir: Path,
    python_exe: str,
    train_rf_script: Path,
    trials_per_fold: int,
    feature_set: str,
) -> Path:
    rf_dir = output_dir / "rf_cv5_matched"
    cmd = [
        python_exe,
        str(train_rf_script),
        "--input", str(matched_csv),
        "--output-dir", str(rf_dir),
        "--seed", str(SEED),
        "--folds", "5",
        "--trials-per-fold", str(trials_per_fold),
        "--feature-set", feature_set,
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return rf_dir / "rf_cv5_oof_predictions.csv"


def train_meta_blender(rf_oof_csv: Path, cnn_matched_csv: Path, output_dir: Path) -> dict:
    rf = pd.read_csv(rf_oof_csv)
    cnn = pd.read_csv(cnn_matched_csv)
    rf["Case ID"] = pd.to_numeric(rf["Case ID"], errors="coerce").astype("Int64")
    cnn["Case ID"] = pd.to_numeric(cnn["Case ID"], errors="coerce").astype("Int64")
    rf["y_true"] = rf["y_true"].map(normalize_label)

    merged = rf.merge(cnn, on="Case ID", how="inner")
    merged = merged.sort_values("Case ID").reset_index(drop=True)

    rf_cols = [f"rf_prob_{label}" for label in LABELS]
    cnn_cols = [f"cnn_prob_{label}" for label in LABELS]
    X = merged[cnn_cols + rf_cols].values
    y = merged["y_true"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(merged), len(LABELS)), dtype=float)
    oof_pred = np.empty(len(merged), dtype=object)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                multi_class="auto",
                max_iter=5000,
                random_state=SEED,
            )),
        ])
        meta.fit(X[train_idx], y[train_idx])
        proba = meta.predict_proba(X[valid_idx])
        classes = meta.named_steps["logistic"].classes_
        aligned = np.zeros((len(valid_idx), len(LABELS)), dtype=float)
        for idx, label in enumerate(classes):
            aligned[:, LABELS.index(label)] = proba[:, idx]
        pred = np.array(LABELS, dtype=object)[aligned.argmax(axis=1)]
        oof_proba[valid_idx] = aligned
        oof_pred[valid_idx] = pred
        metrics = evaluate(y[valid_idx], pred)
        fold_rows.append({"fold": fold, "n_valid": len(valid_idx), **metrics})
        joblib.dump(meta, output_dir / f"meta_blender_fold_{fold}.joblib")
        print(f"[meta fold {fold}/5] macro_f1={metrics['macro_f1']:.4f} balanced_acc={metrics['balanced_accuracy']:.4f}")

    final_meta = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            multi_class="auto",
            max_iter=5000,
            random_state=SEED,
        )),
    ])
    final_meta.fit(X, y)
    joblib.dump(final_meta, output_dir / "meta_blender_final.joblib")

    for idx, label in enumerate(LABELS):
        merged[f"meta_prob_{label}"] = oof_proba[:, idx]
    merged["meta_pred"] = oof_pred
    metrics = evaluate(y, oof_pred)
    cm = pd.DataFrame(
        confusion_matrix(y, oof_pred, labels=LABELS),
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    report = classification_report(y, oof_pred, labels=LABELS, zero_division=0)

    merged.to_csv(output_dir / "meta_blender_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "meta_blender_fold_metrics.csv", index=False)
    cm.to_csv(output_dir / "meta_blender_confusion_matrix.csv")
    (output_dir / "meta_blender_classification_report.txt").write_text(report, encoding="utf-8")
    summary = {
        "seed": SEED,
        "n_rows": int(len(merged)),
        "labels": LABELS,
        "metrics": metrics,
        "rf_oof_csv": str(rf_oof_csv),
        "cnn_matched_csv": str(cnn_matched_csv),
    }
    (output_dir / "meta_blender_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Image-matched clinical RF + CNN/RF meta blender CV=5.")
    parser.add_argument(
        "--clinical-csv",
        default=str(repo_dir / "outputs" / "clinical_pipeline" / "clinical_merged_with_diag_features.csv"),
    )
    parser.add_argument(
        "--cnn-probs",
        default=str(repo_dir / "TMJ_clas" / "clas_runs" / "blend_resnets_optuna_cv5_verbose" / "trial_020" / "meta_oof_predictions.csv"),
    )
    parser.add_argument(
        "--train-rf-script",
        default=str(repo_dir / "TMD_DiagnosisTool" / "train_rf_3class_cv5_optuna.py"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_dir / "outputs" / "matched_cnn_rf_requested_clinical_blender_cv5"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trials-per-fold", type=int, default=5)
    parser.add_argument("--feature-set", default="requested_clinical")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched, matched_cnn = build_matched_clinical(Path(args.clinical_csv), Path(args.cnn_probs), output_dir)

    print("Matched dataset:")
    print(matched["severity"].value_counts().to_string())
    print(f"n={len(matched)}")

    rf_oof = run_rf(
        matched_csv=output_dir / "clinical_image_matched_only.csv",
        output_dir=output_dir,
        python_exe=args.python,
        train_rf_script=Path(args.train_rf_script),
        trials_per_fold=args.trials_per_fold,
        feature_set=args.feature_set,
    )
    summary = train_meta_blender(rf_oof, output_dir / "cnn_probs_image_matched_only.csv", output_dir)
    print("\nMeta blender OOF metrics:")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
