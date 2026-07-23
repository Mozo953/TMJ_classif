from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


CHECKS = [
    (
        "Clinical image dataset",
        ROOT / "TMJ_clas",
        "Required for segmentation training/inference. Expected class folders: normal, mild, severe.",
    ),
    (
        "Old 150 raw OPG dataset for DINOv2",
        ROOT / "TMJ_clas_old150_images_for_dinov3",
        "Optional default for main_dinov2.py. If missing, main_dinov2.py falls back to TMJ_clas.",
    ),
    (
        "Historical post-processed masks",
        ROOT / "TMJ_clas" / "pred_fold02_fossa_erosion_top2_largest" / "masks",
        "Optional default for main_resnet14.py. If missing, run main_resnet14.py --generate-masks-if-missing.",
    ),
    (
        "UNet++ segmentation checkpoint fold 2",
        ROOT / "TMJ_clas" / "unet_runs" / "unetpp_cv5_boundary_fossa_threshold" / "fold_02" / "best_tmj_unetpp_boundary_fossa_threshold.pt",
        "Optional for fast segmentation inference. If missing, main_segmentation.py trains it automatically by default.",
    ),
    (
        "Tracked ResNet14/RF reference probabilities",
        ROOT / "best_models" / "resnet14_plus_tabular_rf" / "resnet14_rf_blend_oof_predictions.csv",
        "Required for fast main_complete.py reference blender.",
    ),
    (
        "ResNet14 params JSON",
        ROOT / "best_models" / "cnn_best" / "base_params.json",
        "Optional. If missing, run_tmj... uses built-in reference params.",
    ),
]


def main() -> None:
    print("TMJ/TMD project reproducibility check")
    print("=" * 42)
    ok_required = True
    for name, path, note in CHECKS:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {name}")
        print(f"  {path}")
        print(f"  {note}")
        if name in {"Clinical image dataset", "Tracked ResNet14/RF reference probabilities"} and not exists:
            ok_required = False
        print()

    print("Recommended commands")
    print("--------------------")
    print("Segmentation only:")
    print("  python main_segmentation.py")
    print("  python main_segmentation.py --no-train-if-missing   # inference-only, requires an existing checkpoint")
    print()
    print("ResNet14:")
    print("  python main_resnet14.py")
    print("  python main_resnet14.py --generate-masks-if-missing")
    print()
    print("DINOv2:")
    print("  python main_dinov2.py")
    print("  python main_dinov2.py --data-dir <dataset-with-normal-mild-severe>")
    print()
    print("Complete reference blender:")
    print("  python main_complete.py")
    print()
    if not ok_required:
        raise SystemExit("Some required files are missing. See messages above.")


if __name__ == "__main__":
    main()
