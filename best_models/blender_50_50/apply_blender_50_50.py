from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ["Mild", "Normal", "Severe"]
CNN_WEIGHT = 0.50
RF_WEIGHT = 0.50


def extract_case_id(value) -> int | None:
    nums = re.findall(r"\d+", str(value))
    return int(nums[-1]) if nums else None


def normalize_probs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)
    sums = values.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return values / sums


def load_cnn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Case ID" not in df.columns:
        if "mask" not in df.columns:
            raise ValueError("CNN file must contain `Case ID` or `mask`.")
        df["Case ID"] = df["mask"].map(extract_case_id)

    rename = {
        "prob_mild": "cnn_prob_Mild",
        "prob_normal": "cnn_prob_Normal",
        "prob_severe": "cnn_prob_Severe",
    }
    df = df.rename(columns=rename)
    cols = ["Case ID"] + [f"cnn_prob_{label}" for label in LABELS]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"CNN file missing columns: {missing}")
    df = df[cols].copy()
    df["Case ID"] = pd.to_numeric(df["Case ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Case ID"]).drop_duplicates("Case ID", keep="first")
    df[[f"cnn_prob_{label}" for label in LABELS]] = normalize_probs(
        df[[f"cnn_prob_{label}" for label in LABELS]].to_numpy()
    )
    return df


def load_rf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["Case ID"] + [f"rf_prob_{label}" for label in LABELS]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"RF file missing columns: {missing}")
    df = df[cols + [col for col in ["y_true", "rf_pred"] if col in df.columns]].copy()
    df["Case ID"] = pd.to_numeric(df["Case ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Case ID"]).drop_duplicates("Case ID", keep="first")
    df[[f"rf_prob_{label}" for label in LABELS]] = normalize_probs(
        df[[f"rf_prob_{label}" for label in LABELS]].to_numpy()
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the validated 50/50 CNN + RF requested clinical blender.")
    parser.add_argument("--cnn-probs", required=True, help="CSV with CNN probabilities.")
    parser.add_argument("--rf-probs", required=True, help="CSV with RF requested clinical probabilities.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    cnn = load_cnn(Path(args.cnn_probs))
    rf = load_rf(Path(args.rf_probs))
    merged = rf.merge(cnn, on="Case ID", how="inner").sort_values("Case ID").reset_index(drop=True)

    rf_probs = merged[[f"rf_prob_{label}" for label in LABELS]].to_numpy()
    cnn_probs = merged[[f"cnn_prob_{label}" for label in LABELS]].to_numpy()
    blend = CNN_WEIGHT * cnn_probs + RF_WEIGHT * rf_probs

    for index, label in enumerate(LABELS):
        merged[f"blend_prob_{label}"] = blend[:, index]
    merged["blend_pred"] = np.array(LABELS, dtype=object)[blend.argmax(axis=1)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Saved {len(merged)} blended predictions to {output}")


if __name__ == "__main__":
    main()

