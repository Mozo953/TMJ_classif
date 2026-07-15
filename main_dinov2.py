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
    parser = argparse.ArgumentParser(description="Launch DINOv2 CV3 on raw OPG images.")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "TMJ_clas_old150_images_for_dinov3"),
        help="Default = frozen old 150-image dataset used for the best previous pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "code_classification" / "dinov2_tmd_experiment" / "outputs_dinov2_cv3_reproduce_old150"),
    )
    parser.add_argument("--model", default="vit_small_patch14_dinov2")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        [
            sys.executable,
            str(ROOT / "code_classification" / "dinov2_tmd_experiment" / "dinov2_tmd_cv.py"),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
            "--model",
            args.model,
            "--image-size",
            str(args.image_size),
            "--folds",
            str(args.folds),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--class-weighted-loss",
            "--device",
            "cuda",
        ]
    )


if __name__ == "__main__":
    main()

