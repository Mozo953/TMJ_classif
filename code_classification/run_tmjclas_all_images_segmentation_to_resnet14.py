from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = PROJECT_ROOT / "code_segmentation"
SEG_EXP = SEG_ROOT / "experimentation"
CLS_ROOT = PROJECT_ROOT / "code_classification"
for path in [str(SEG_ROOT), str(SEG_EXP), str(CLS_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from infer_cv2_with_fossa_erosion_grid import (  # noqa: E402
    checkpoint_path,
    erode_fossa,
    load_model,
    pred_from_probs,
    predict_probs,
    save_palette_mask,
)
from keep_top2_mean_probability_components import (  # noqa: E402
    filter_mask_top2,
)
from clas_blend_optuna_cv3_Best_MODEL import (  # noqa: E402
    CLASS_NAMES,
    MaskSample,
    make_loader,
    make_model,
    predict_proba,
    seed_everything,
    train_base_model,
    write_predictions,
)
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402
from clas_blend_optuna_cv3_Best_MODEL import MaskDataset  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DISPLAY_ORDER = ["normal", "mild", "severe"]
DEFAULT_RESNET14_PARAMS = {
    "epochs": 45,
    "image_size": 160,
    "batch_size": 4,
    "base_channels": 16,
    "lr": 0.001498958154469183,
    "weight_decay": 0.0002413655863489142,
    "dropout": 0.07864781233782388,
    "label_smoothing": 0.04306671235119111,
}
PRED_COLORS = {
    1: (255, 0, 0, 175),      # condyle
    2: (255, 150, 0, 195),    # glenoid fossa
}


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    severity: str


def case_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"case\s*no\.\s*(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), path.name.lower()
    return 10**9, path.name.lower()


def collect_image_samples(data_root: Path) -> list[ImageSample]:
    samples: list[ImageSample] = []
    for severity in CLASS_NAMES:
        class_dir = data_root / severity
        if not class_dir.exists():
            raise RuntimeError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir(), key=case_sort_key):
            if path.name.startswith("._") or path.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append(ImageSample(path.resolve(), severity))
    if not samples:
        raise RuntimeError(f"No images found in {data_root}")
    return samples


def make_prediction_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = Image.fromarray(image).convert("RGBA")
    layer = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    for class_id, color in PRED_COLORS.items():
        layer[mask == class_id] = color
    result = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    labels = [("Pred condyle", PRED_COLORS[1]), ("Pred fossa", PRED_COLORS[2])]
    x, y = 8, 8
    draw.rounded_rectangle((x, y, x + 135, y + 58), radius=6, fill=(0, 0, 0, 165))
    for i, (label, color) in enumerate(labels):
        yy = y + 8 + i * 22
        draw.rectangle((x + 8, yy + 3, x + 23, yy + 18), fill=color)
        draw.text((x + 30, yy + 2), label, fill=(255, 255, 255, 255), font=font)
    return result


def make_contact_sheets(output_dir: Path) -> None:
    font = ImageFont.load_default()
    overlays_root = output_dir / "overlays"
    for severity_dir in sorted([p for p in overlays_root.iterdir() if p.is_dir()] if overlays_root.exists() else []):
        files = sorted(severity_dir.glob("*_overlay.png"), key=case_sort_key)[:24]
        if not files:
            continue
        thumb_w, thumb_h = 340, 205
        label_h = 20
        cols = 3
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(files):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (i % cols) * thumb_w + (thumb_w - image.width) // 2
            y = (i // cols) * (thumb_h + label_h)
            sheet.paste(image, (x, y))
            draw.text(((i % cols) * thumb_w + 6, y + thumb_h + 2), path.stem.replace("_overlay", ""), fill=(0, 0, 0), font=font)
        sheet.save(output_dir / f"contact_sheet_{severity_dir.name}.jpg", quality=92)


def run_segmentation(args: argparse.Namespace) -> Path:
    samples = collect_image_samples(args.data_root)
    output_dir = args.seg_output_dir
    masks_root = output_dir / "masks"
    overlays_root = output_dir / "overlays"
    masks_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint_path(args.seg_run_dir, args.inference_fold)
    if not ckpt_path.exists():
        if args.train_segmentation_if_missing:
            print(
                f"[seg] Missing checkpoint: {ckpt_path}\n"
                f"[seg] Training segmentation run first into: {args.seg_run_dir}",
                flush=True,
            )
            train_cmd = [
                sys.executable,
                str(SEG_ROOT / "seg_cv_best.py"),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(args.seg_run_dir),
                "--folds",
                str(args.seg_train_folds),
                "--epochs",
                str(args.seg_train_epochs),
                "--batch-size",
                str(args.seg_train_batch_size),
                "--image-size",
                str(args.seg_image_size),
                "--base-channels",
                str(args.seg_train_base_channels),
                "--line-thickness",
                str(args.seg_train_line_thickness),
                "--crop-x-fraction",
                str(args.crop_x_fraction),
                "--crop-y-fraction",
                str(args.crop_y_fraction),
                "--lr",
                str(args.seg_train_lr),
                "--dice-loss-weight",
                str(args.seg_train_dice_loss_weight),
                "--boundary-loss-weight",
                str(args.seg_train_boundary_loss_weight),
                "--seed",
                str(args.seed),
                "--num-workers",
                str(args.num_workers),
            ]
            if args.seg_train_fixed_fossa_threshold is not None:
                train_cmd.extend(["--fixed-fossa-threshold", str(args.seg_train_fixed_fossa_threshold)])
            if args.seg_train_split_source_dir is not None:
                train_cmd.extend(["--split-source-dir", str(args.seg_train_split_source_dir)])
            print("$ " + " ".join(train_cmd), flush=True)
            subprocess.run(train_cmd, cwd=str(PROJECT_ROOT), check=True)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                "Segmentation checkpoint is missing.\n"
                f"Expected checkpoint:\n  {ckpt_path}\n\n"
                "Fix options:\n"
                "  1) Train it first:\n"
                "     python main_segmentation.py --train-if-missing\n\n"
                "  2) Or pass an existing run dir:\n"
                "     python main_segmentation.py --seg-run-dir <path-to-unetpp-run>\n\n"
                "  3) Or if you already have masks, skip segmentation in the full pipeline:\n"
                "     python code_classification/run_tmjclas_all_images_segmentation_to_resnet14.py --skip-segmentation --seg-output-dir <dir-containing-masks>\n"
            )
    model, checkpoint = load_model(ckpt_path, device)
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", args.seg_image_size))
    fossa_threshold = float(checkpoint.get("best_fossa_threshold", args.fossa_threshold))
    x_fraction = float(ckpt_args.get("crop_x_fraction", args.crop_x_fraction))
    y_fraction = float(ckpt_args.get("crop_y_fraction", args.crop_y_fraction))

    rows = []
    component_rows = []
    print(
        f"[seg] images={len(samples)} fold={args.inference_fold} device={device} "
        f"threshold={fossa_threshold} erosion={args.fossa_erosion_kernel} rank_by={args.component_rank_by}",
        flush=True,
    )
    for idx, sample in enumerate(samples, start=1):
        if idx == 1 or idx % 25 == 0:
            print(f"[seg] {idx}/{len(samples)} {sample.severity}/{sample.image_path.name}", flush=True)
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        probs = predict_probs(model, image, image_size, device, x_fraction, y_fraction)
        raw_mask = pred_from_probs(probs, fossa_threshold, x_fraction, y_fraction)
        raw_mask = erode_fossa(raw_mask, args.fossa_erosion_kernel)
        filtered_mask, components = filter_mask_top2(
            raw_mask,
            probs,
            max_components=args.max_components,
            rank_by=args.component_rank_by,
        )

        mask_dir = masks_root / sample.severity
        overlay_dir = overlays_root / sample.severity
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{sample.image_path.stem}_mask.png"
        overlay_path = overlay_dir / f"{sample.image_path.stem}_overlay.png"
        save_palette_mask(filtered_mask, mask_path)
        make_prediction_overlay(image, filtered_mask).convert("RGB").save(overlay_path, quality=95)

        counts = np.bincount(filtered_mask.reshape(-1), minlength=3)
        rows.append(
            {
                "severity": sample.severity,
                "image": str(sample.image_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "fossa_threshold": fossa_threshold,
                "fossa_erosion_kernel": args.fossa_erosion_kernel,
                "max_components_per_class": args.max_components,
                "component_rank_by": args.component_rank_by,
                "background_pixels": int(counts[0]),
                "condyle_pixels": int(counts[1]),
                "glenoid_fossa_pixels": int(counts[2]),
            }
        )
        for component in components:
            component_rows.append(
                {
                    "severity": sample.severity,
                    "image": str(sample.image_path),
                    "class_id": component["class_id"],
                    "class_name": component["class_name"],
                    "component_id": component["component_id"],
                    "area_pixels": component["area_pixels"],
                    "prob_sum": component["prob_sum"],
                    "prob_mean": component["prob_mean"],
                    "prob_max": component["prob_max"],
                    "keep": component["keep"],
                }
            )

    with (output_dir / "prediction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "component_selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        writer.writerows(component_rows)
    make_contact_sheets(output_dir)
    return masks_root


def collect_mask_samples(mask_root: Path) -> list[MaskSample]:
    samples: list[MaskSample] = []
    for label_idx, label_name in enumerate(CLASS_NAMES):
        class_dir = mask_root / label_name
        for mask_path in sorted(class_dir.glob("*_mask.png"), key=case_sort_key):
            samples.append(MaskSample(mask_path=mask_path, label=label_idx, label_name=label_name))
    if not samples:
        raise RuntimeError(f"No masks found in {mask_root}")
    return samples


def metric_dict(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    cm_native = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    display_indices = [CLASS_NAMES.index(name) for name in DISPLAY_ORDER]
    cm_display = confusion_matrix(y_true, pred, labels=display_indices)
    recalls = recall_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "class_order_native": CLASS_NAMES,
        "class_order_display": DISPLAY_ORDER,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "recall_by_class": {name: float(recalls[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "confusion_matrix_native": cm_native.tolist(),
        "confusion_matrix_display_normal_mild_severe": cm_display.tolist(),
    }


def make_balanced_train_loader(samples: list[MaskSample], image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    sample_weights = np.array([1.0 / max(counts[label], 1.0) for label in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )
    ds = MaskDataset(samples, image_size=image_size, train=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def balanced_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_base_model_balanced(
    model_name: str,
    train_samples: list[MaskSample],
    valid_samples: list[MaskSample],
    args: argparse.Namespace,
    params: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> np.ndarray:
    verbose = getattr(args, "verbose", False)
    if verbose:
        print(
            f"[base-balanced] model={model_name} train={len(train_samples)} valid={len(valid_samples)} "
            f"epochs={params['epochs']} image_size={params['image_size']} batch={params['batch_size']}",
            flush=True,
        )

    model = make_model(model_name, params["base_channels"], params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    train_labels = np.array([sample.label for sample in train_samples], dtype=np.int64)
    class_weights = balanced_class_weights(train_labels, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=params["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = make_balanced_train_loader(
        train_samples,
        params["image_size"],
        params["batch_size"],
        args.num_workers,
    )
    valid_loader = make_loader(valid_samples, params["image_size"], params["batch_size"], False, args.num_workers)

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        probs, y_true = predict_proba(model, valid_loader, device)
        y_pred = probs.argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch == params["epochs"] or epoch % max(1, params["epochs"] // 5) == 0):
            print(
                f"[base-balanced] model={model_name} epoch={epoch:03d}/{params['epochs']} "
                f"loss={running_loss / max(seen, 1):.4f} valid_macro_f1={f1:.4f} best={best_f1:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("No best state saved")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state": best_state,
            "params": params,
            "class_names": CLASS_NAMES,
            "balanced_training": True,
            "class_weights": class_weights.detach().cpu().numpy().tolist(),
        },
        checkpoint_path,
    )
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probs, _ = predict_proba(model, valid_loader, device)
    return probs


def run_resnet14_cv(args: argparse.Namespace, mask_root: Path) -> Path:
    if args.base_params_path.exists():
        with args.base_params_path.open("r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = DEFAULT_RESNET14_PARAMS.copy()
        print(
            f"[cls] base params file not found: {args.base_params_path}\n"
            "[cls] using built-in ResNet14 reference parameters instead.",
            flush=True,
        )
    if args.epochs is not None:
        params["epochs"] = args.epochs
    params["image_size"] = args.classification_image_size
    params["batch_size"] = args.batch_size

    output_dir = args.classification_output_dir
    model_dir = output_dir / "resnet14"
    model_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_mask_samples(mask_root)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    counts = {name: int((labels == idx).sum()) for idx, name in enumerate(CLASS_NAMES)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cls] ResNet14 CV={args.folds} samples={len(samples)} counts={counts} device={device}", flush=True)
    print(f"[cls] params={params}", flush=True)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(samples), len(CLASS_NAMES)), dtype=np.float32)
    oof_folds = np.full(len(samples), -1, dtype=np.int64)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.arange(len(samples)), labels), start=1):
        print(f"[cls] fold={fold}/{args.folds} train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        train_samples = [samples[i] for i in train_idx]
        valid_samples = [samples[i] for i in valid_idx]
        ckpt = model_dir / "checkpoints" / f"fold_{fold:02d}.pt"
        if args.balanced_training:
            oof[valid_idx] = train_base_model_balanced("resnet14", train_samples, valid_samples, args, params, device, ckpt)
        else:
            oof[valid_idx] = train_base_model("resnet14", train_samples, valid_samples, args, params, device, ckpt)
        oof_folds[valid_idx] = fold
        fold_metrics = metric_dict(labels[valid_idx], oof[valid_idx])
        with (model_dir / f"fold_{fold:02d}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2)
        fold_rows.append({"fold": fold, **{k: fold_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_recall"]}})
        print(f"[cls] fold={fold} acc={fold_metrics['accuracy']:.4f} macro_f1={fold_metrics['macro_f1']:.4f}", flush=True)

    metrics = metric_dict(labels, oof)
    np.save(model_dir / "oof_resnet14.npy", oof)
    np.save(model_dir / "oof_folds.npy", oof_folds)
    write_predictions(model_dir / "oof_predictions.csv", samples, labels, oof)
    with (model_dir / "oof_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (model_dir / "fold_metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (model_dir / "confusion_matrix_normal_mild_severe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *DISPLAY_ORDER])
        for name, row in zip(DISPLAY_ORDER, metrics["confusion_matrix_display_normal_mild_severe"]):
            writer.writerow([name, *row])

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": str(args.data_root),
                "seg_output_dir": str(args.seg_output_dir),
                "mask_root": str(mask_root),
                "classification_output_dir": str(output_dir),
                "class_counts": counts,
                "folds": args.folds,
                "seed": args.seed,
                "params": params,
                "metrics": metrics,
                "balanced_training": args.balanced_training,
            },
            f,
            indent=2,
        )
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-images TMJ segmentation, then ResNet14 CV3 OOF classification.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "TMJ_clas")
    parser.add_argument("--seg-run-dir", type=Path, default=PROJECT_ROOT / "TMJ_clas" / "unet_runs" / "unetpp_cv5_boundary_fossa_threshold")
    parser.add_argument("--seg-output-dir", type=Path, default=PROJECT_ROOT / "TMJ_clas" / "pred_all_images_fold02_fossa_erosion_top2_largest")
    parser.add_argument("--classification-output-dir", type=Path, default=PROJECT_ROOT / "best_models" / "tmjclas_all_images_seg_to_resnet14")
    parser.add_argument("--base-params-path", type=Path, default=PROJECT_ROOT / "best_models" / "cnn_best" / "base_params.json")
    parser.add_argument("--inference-fold", type=int, default=2)
    parser.add_argument("--seg-image-size", type=int, default=512)
    parser.add_argument("--crop-x-fraction", type=float, default=0.38)
    parser.add_argument("--crop-y-fraction", type=float, default=0.58)
    parser.add_argument("--fossa-threshold", type=float, default=0.95)
    parser.add_argument("--fossa-erosion-kernel", type=int, default=1)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--component-rank-by", choices=["area", "mean"], default="area")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classification-image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--balanced-training", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--train-segmentation-if-missing",
        action="store_true",
        help="If the requested UNet++ checkpoint is missing, train code_segmentation/seg_cv_best.py before inference.",
    )
    parser.add_argument("--seg-train-folds", type=int, default=5)
    parser.add_argument("--seg-train-epochs", type=int, default=50)
    parser.add_argument("--seg-train-batch-size", type=int, default=2)
    parser.add_argument("--seg-train-base-channels", type=int, default=16)
    parser.add_argument("--seg-train-line-thickness", type=int, default=11)
    parser.add_argument("--seg-train-lr", type=float, default=1e-3)
    parser.add_argument("--seg-train-dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--seg-train-boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--seg-train-fixed-fossa-threshold", type=float, default=None)
    parser.add_argument("--seg-train-split-source-dir", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.skip_segmentation:
        mask_root = args.seg_output_dir / "masks"
    else:
        mask_root = run_segmentation(args)
    if not args.skip_classification:
        model_dir = run_resnet14_cv(args, mask_root)
        print(f"[done] ResNet14 results: {model_dir}", flush=True)
    else:
        print(f"[done] segmentation masks: {mask_root}", flush=True)


if __name__ == "__main__":
    main()
