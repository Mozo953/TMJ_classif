from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the reference ResNet14 mask classifier.")
    parser.add_argument(
        "--mask-run-dir",
        default=str(ROOT / "TMJ_clas" / "pred_fold02_fossa_erosion_top2_largest"),
        help="Directory containing a masks/ subfolder. Default = old/best 150-case post-processed segmentation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "best_models" / "reproduce_resnet14_old150"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--balanced-training", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "code_classification" / "run_tmjclas_all_images_segmentation_to_resnet14.py"),
        "--skip-segmentation",
        "--seg-output-dir",
        args.mask_run_dir,
        "--classification-output-dir",
        args.output_dir,
        "--seed",
        str(args.seed),
        "--folds",
        str(args.folds),
        "--epochs",
        str(args.epochs),
        "--verbose",
    ]
    if args.balanced_training:
        cmd.append("--balanced-training")
    run(cmd)


if __name__ == "__main__":
    main()

