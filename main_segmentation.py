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
    parser = argparse.ArgumentParser(description="Launch TMJ segmentation only.")
    parser.add_argument("--data-root", default=str(ROOT / "TMJ_clas"))
    parser.add_argument(
        "--seg-run-dir",
        default=str(ROOT / "TMJ_clas" / "unet_runs" / "unetpp_cv5_boundary_fossa_threshold"),
        help="UNet++ training run directory containing fold_XX/best_tmj_unetpp_boundary_fossa_threshold.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "TMJ_clas" / "pred_all_images_fold02_fossa_erosion_top2_largest"),
    )
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train the UNet++ boundary/fossa-threshold segmentation model if the checkpoint is missing.",
    )
    parser.add_argument("--train-folds", type=int, default=5)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=2)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "code_classification" / "run_tmjclas_all_images_segmentation_to_resnet14.py"),
        "--data-root",
        args.data_root,
        "--seg-run-dir",
        args.seg_run_dir,
        "--seg-output-dir",
        args.output_dir,
        "--inference-fold",
        str(args.fold),
        "--skip-classification",
        "--seg-train-folds",
        str(args.train_folds),
        "--seg-train-epochs",
        str(args.train_epochs),
        "--seg-train-batch-size",
        str(args.train_batch_size),
    ]
    if args.train_if_missing:
        cmd.append("--train-segmentation-if-missing")
    run(cmd)


if __name__ == "__main__":
    main()
