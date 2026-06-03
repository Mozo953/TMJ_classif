"""
Run the full Excel preprocessing pipeline and the RF 3-class CV=5 model.

This is the clean one-command entry point for:
    normal_clinical(1).xlsx + clinicaldataa.csv -> merged features -> RF model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
REPO_DIR = PROJECT_DIR.parent
CODEX_PYTHON = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Full clinical preprocessing + RF runner.")
    parser.add_argument(
        "--normal-xlsx",
        default=r"C:\Users\sadmin\Downloads\normal_clinical(1).xlsx",
        help="Raw normal clinical Excel file.",
    )
    parser.add_argument(
        "--tmd-csv",
        default=str(PROJECT_DIR / "clinicaldataa.csv"),
        help="Structured TMD clinical CSV.",
    )
    parser.add_argument(
        "--clinical-output-dir",
        default=str(PROJECT_DIR / "outputs" / "clinical_pipeline"),
        help="Output dir for merged clinical features.",
    )
    parser.add_argument(
        "--rf-output-dir",
        default=str(PROJECT_DIR / "outputs" / "rf_3class_cv5_optuna"),
        help="Output dir for RF model and CV outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials-per-fold", type=int, default=5)
    parser.add_argument(
        "--preprocess-python",
        default=str(CODEX_PYTHON if CODEX_PYTHON.exists() else Path(sys.executable)),
        help="Python executable with pandas+openpyxl for Excel preprocessing.",
    )
    parser.add_argument(
        "--ml-python",
        default=sys.executable,
        help="Python executable with sklearn+optuna for RF training.",
    )
    args = parser.parse_args()

    normal_xlsx = require_file(args.normal_xlsx, "Normal clinical Excel")
    tmd_csv = require_file(args.tmd_csv, "TMD clinical CSV")
    print("Input check OK")
    print(f"Normal Excel: {normal_xlsx}")
    print(f"TMD CSV: {tmd_csv}")

    run([
        args.preprocess_python,
        str(PROJECT_DIR / "run_full_clinical_pipeline.py"),
        "--normal-xlsx", str(normal_xlsx),
        "--tmd-csv", str(tmd_csv),
        "--output-dir", args.clinical_output_dir,
    ], cwd=PROJECT_DIR)

    merged_path = Path(args.clinical_output_dir) / "clinical_merged_with_diag_features.csv"
    run([
        args.ml_python,
        str(PROJECT_DIR / "train_rf_3class_cv5_optuna.py"),
        "--input", str(merged_path),
        "--output-dir", args.rf_output_dir,
        "--seed", str(args.seed),
        "--folds", str(args.folds),
        "--trials-per-fold", str(args.trials_per_fold),
    ], cwd=PROJECT_DIR)

    print("\nDone.")
    print(f"Merged clinical features: {merged_path}")
    print(f"RF outputs: {args.rf_output_dir}")


if __name__ == "__main__":
    main()
