from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


CLASS_NAMES = ["mild", "normal", "severe"]
BASE_MODEL_NAMES = ["resnet14", "resnet20", "resnet26", "small_cnn", "depthwise_cnn"]


def load_labels(dataset_csv: Path) -> np.ndarray:
    labels = []
    label_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    with dataset_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(label_to_id[row["label"]])
    return np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OOF metrics for the 5 base models in the best Optuna blend trial.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\blend_resnets_optuna_cv3_top2_largest"),
    )
    parser.add_argument("--trial-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.trial_dir is None:
        best_json = args.run_dir / "best_trial.json"
        with best_json.open("r", encoding="utf-8") as f:
            best = json.load(f)
        trial_dir = Path(best["user_attrs"]["trial_dir"])
    else:
        trial_dir = args.trial_dir

    y_true = load_labels(args.run_dir / "dataset.csv")
    rows = []
    for model_name in BASE_MODEL_NAMES:
        oof_path = trial_dir / f"oof_{model_name}.npy"
        probs = np.load(oof_path)
        pred = probs.argmax(axis=1)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_true, pred),
                "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
                "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
                "precision_mild": precision_score(y_true, pred, labels=[0], average="macro", zero_division=0),
                "precision_normal": precision_score(y_true, pred, labels=[1], average="macro", zero_division=0),
                "precision_severe": precision_score(y_true, pred, labels=[2], average="macro", zero_division=0),
            }
        )

        cm_path = trial_dir / f"confusion_{model_name}.csv"
        with cm_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *CLASS_NAMES])
            for class_name, cm_row in zip(CLASS_NAMES, confusion_matrix(y_true, pred, labels=[0, 1, 2])):
                writer.writerow([class_name, *cm_row.tolist()])

    out_path = trial_dir / "base_model_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Trial dir: {trial_dir}")
    print(f"Saved: {out_path}")
    for row in sorted(rows, key=lambda item: item["macro_f1"], reverse=True):
        print(
            f"{row['model']}: macro_precision={row['macro_precision']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} accuracy={row['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
