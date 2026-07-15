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
        "--output-dir",
        default=str(ROOT / "TMJ_clas" / "pred_all_images_fold02_fossa_erosion_top2_largest"),
    )
    parser.add_argument("--fold", type=int, default=2)
    args = parser.parse_args()

    run(
        [
            sys.executable,
            str(ROOT / "code_classification" / "run_tmjclas_all_images_segmentation_to_resnet14.py"),
            "--data-root",
            args.data_root,
            "--seg-output-dir",
            args.output_dir,
            "--inference-fold",
            str(args.fold),
            "--skip-classification",
        ]
    )


if __name__ == "__main__":
    main()

