from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS = ["Mild", "Normal", "Severe"]
IMAGE_WEIGHT = 0.70
RF_WEIGHT = 0.30


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def normalize_probs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)
    sums = values.sum(axis=1, keepdims=True)
    sums[sums == 0.0] = 1.0
    return values / sums


def load_image_probs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Supported sources:
    # 1) ResNet14+RF table: Case ID + resnet14_prob_Mild/Normal/Severe
    # 2) ResNet14 OOF table: mask + prob_mild/prob_normal/prob_severe
    # 3) Legacy CNN table: Case ID + cnn_prob_Mild/Normal/Severe
    if "Case ID" not in df.columns:
        if "mask" in df.columns:
            df["Case ID"] = df["mask"].astype(str).str.extract(r"(\d+)").astype(float)
        elif "case_id" in df.columns:
            df["Case ID"] = df["case_id"]
        else:
            raise ValueError(f"Image probability file has no Case ID/case_id/mask column: {path}")

    rename = {
        "prob_mild": "image_prob_Mild",
        "prob_normal": "image_prob_Normal",
        "prob_severe": "image_prob_Severe",
        "cnn_prob_Mild": "image_prob_Mild",
        "cnn_prob_Normal": "image_prob_Normal",
        "cnn_prob_Severe": "image_prob_Severe",
        "resnet14_prob_Mild": "image_prob_Mild",
        "resnet14_prob_Normal": "image_prob_Normal",
        "resnet14_prob_Severe": "image_prob_Severe",
    }
    df = df.rename(columns=rename)
    cols = ["Case ID"] + [f"image_prob_{label}" for label in LABELS]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Image probability file missing columns: {missing}")
    out = df[cols].copy()
    out["Case ID"] = pd.to_numeric(out["Case ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Case ID"]).drop_duplicates("Case ID", keep="first")
    out[[f"image_prob_{label}" for label in LABELS]] = normalize_probs(
        out[[f"image_prob_{label}" for label in LABELS]].to_numpy()
    )
    return out


def load_rf_probs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["Case ID"] + [f"rf_prob_{label}" for label in LABELS]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"RF probability file missing columns: {missing}")
    keep = cols + [col for col in ["y_true", "rf_pred"] if col in df.columns]
    out = df[keep].copy()
    out["Case ID"] = pd.to_numeric(out["Case ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Case ID"]).drop_duplicates("Case ID", keep="first")
    out[[f"rf_prob_{label}" for label in LABELS]] = normalize_probs(
        out[[f"rf_prob_{label}" for label in LABELS]].to_numpy()
    )
    return out


def apply_weighted_blender(image_probs_path: Path, rf_probs_path: Path, output_path: Path) -> None:
    image_df = load_image_probs(image_probs_path)
    rf_df = load_rf_probs(rf_probs_path)
    merged = rf_df.merge(image_df, on="Case ID", how="inner").sort_values("Case ID").reset_index(drop=True)

    image_probs = merged[[f"image_prob_{label}" for label in LABELS]].to_numpy()
    rf_probs = merged[[f"rf_prob_{label}" for label in LABELS]].to_numpy()
    blend = IMAGE_WEIGHT * image_probs + RF_WEIGHT * rf_probs

    for index, label in enumerate(LABELS):
        merged[f"blend_prob_{label}"] = blend[:, index]
    merged["blend_pred"] = np.array(LABELS, dtype=object)[blend.argmax(axis=1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} blended predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complete reference launcher: segmentation masks + ResNet14 probabilities + "
            "RF tabular probabilities + probability blender."
        )
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "main_complete_reference_blender"))
    parser.add_argument(
        "--cnn-probs",
        default=str(ROOT / "best_models" / "resnet14_plus_tabular_rf" / "resnet14_rf_blend_oof_predictions.csv"),
        help="Default = stored ResNet14 probabilities from the old 150-case pipeline.",
    )
    parser.add_argument(
        "--rf-probs",
        default=str(ROOT / "best_models" / "rf_requested_clinical" / "rf_cv5_oof_predictions.csv"),
    )
    parser.add_argument(
        "--retrain-resnet14",
        action="store_true",
        help="Retrain ResNet14 first, then blend its new OOF probabilities with the stored RF OOF predictions.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cnn_probs = Path(args.cnn_probs)
    rf_probs = Path(args.rf_probs)
    if args.retrain_resnet14:
        resnet_out = output_dir / "resnet14_retrained"
        run(
            [
                sys.executable,
                str(ROOT / "main_resnet14.py"),
                "--output-dir",
                str(resnet_out),
            ]
        )
        cnn_probs = resnet_out / "resnet14" / "oof_predictions.csv"
    if not cnn_probs.exists():
        raise FileNotFoundError(
            "Image/CNN probability file is missing.\n"
            f"Expected:\n  {cnn_probs}\n\n"
            "Fix options:\n"
            "  1) Run ResNet14 first:\n"
            "     python main_complete.py --retrain-resnet14\n\n"
            "  2) Or pass an existing probability CSV:\n"
            "     python main_complete.py --cnn-probs <path>\n"
        )
    if not rf_probs.exists():
        probe = pd.read_csv(cnn_probs, nrows=1)
        has_rf_columns = all(f"rf_prob_{label}" in probe.columns for label in LABELS)
        if has_rf_columns:
            print(
                f"[complete] RF probability file not found: {rf_probs}\n"
                f"[complete] using RF columns already present in: {cnn_probs}",
                flush=True,
            )
            rf_probs = cnn_probs
        else:
            raise FileNotFoundError(
                "RF probability file is missing.\n"
                f"Expected:\n  {rf_probs}\n\n"
                "Pass a RF OOF probability CSV containing Case ID + rf_prob_Mild/rf_prob_Normal/rf_prob_Severe:\n"
                "  python main_complete.py --rf-probs <path>\n"
            )

    apply_weighted_blender(
        image_probs_path=cnn_probs,
        rf_probs_path=rf_probs,
        output_path=output_dir / "final_blender_predictions.csv",
    )

    print("\nReference metrics from the validated saved pipeline:")
    print("  ResNet14 alone accuracy: 0.6867")
    print("  ResNet14 + RF blender weights: 0.70 image / 0.30 RF")
    print("  ResNet14 + RF 70/30 blender accuracy: 0.7533")
    print(f"  Output: {output_dir / 'final_blender_predictions.csv'}")


if __name__ == "__main__":
    main()
