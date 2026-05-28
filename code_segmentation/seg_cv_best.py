from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_ARCHIVE = PROJECT_ROOT / "code_archive_seg"
for path in [PROJECT_ROOT, SEG_ARCHIVE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_tmj_unetpp_cv_precrop import (
    CLASS_NAMES,
    DiceLoss,
    Sample,
    TMJPreCropDataset,
    UNetPlusPlus,
    collect_samples,
    seed_everything,
    train_transform,
    valid_transform,
    write_metrics_csv,
)


FOSSA_CLASS_ID = 2
FOSSA_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]


class BoundaryLoss(nn.Module):
    """Boundary loss on foreground classes using signed distance maps."""

    def __init__(self, num_classes: int, ignore_background: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    @staticmethod
    def signed_distance(mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)
        if not mask.any():
            return np.ones(mask.shape, dtype=np.float32)
        if mask.all():
            return -np.ones(mask.shape, dtype=np.float32)
        pos_dist = distance_transform_edt(mask)
        neg_dist = distance_transform_edt(~mask)
        sdf = neg_dist - pos_dist
        max_abs = np.max(np.abs(sdf))
        if max_abs > 0:
            sdf = sdf / max_abs
        return sdf.astype(np.float32)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        start = 1 if self.ignore_background else 0
        target_np = target.detach().cpu().numpy()
        distance_maps = []
        for batch_idx in range(target_np.shape[0]):
            class_maps = []
            for class_idx in range(start, self.num_classes):
                class_maps.append(self.signed_distance(target_np[batch_idx] == class_idx))
            distance_maps.append(np.stack(class_maps, axis=0))
        dist = torch.from_numpy(np.stack(distance_maps, axis=0)).to(device=logits.device, dtype=logits.dtype)
        return torch.mean(probs[:, start:] * dist)


def write_split(path: Path, train_samples: list[Sample], valid_samples: list[Sample], test_samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "severity", "image", "json"])
        for split, split_samples in [("train", train_samples), ("valid", valid_samples), ("test", test_samples)]:
            for sample in split_samples:
                writer.writerow([split, sample.severity, str(sample.image_path), str(sample.json_path)])


def read_split(path: Path, samples_by_image: dict[str, Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    splits = {"train": [], "valid": [], "test": []}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample = samples_by_image.get(str(Path(row["image"])))
            if sample is None:
                raise RuntimeError(f"Split references an unknown image: {row['image']}")
            splits[row["split"]].append(sample)
    return splits["train"], splits["valid"], splits["test"]


def make_loader(samples: list[Sample], args: argparse.Namespace, train: bool) -> DataLoader:
    transform = train_transform(args.image_size) if train else valid_transform(args.image_size)
    dataset = TMJPreCropDataset(
        samples,
        args.image_size,
        args.line_thickness,
        args.crop_x_fraction,
        args.crop_y_fraction,
        transform,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)


def prediction_from_logits(logits: torch.Tensor, fossa_threshold: float | None = None) -> torch.Tensor:
    if fossa_threshold is None:
        return torch.argmax(logits, dim=1)

    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(logits[:, :FOSSA_CLASS_ID], dim=1)
    pred = pred.to(torch.long)
    pred[probs[:, FOSSA_CLASS_ID] > fossa_threshold] = FOSSA_CLASS_ID
    return pred


def metrics_from_prediction(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, np.ndarray]:
    totals = {
        "intersection": np.zeros(num_classes, dtype=np.float64),
        "pred_pixels": np.zeros(num_classes, dtype=np.float64),
        "target_pixels": np.zeros(num_classes, dtype=np.float64),
        "union": np.zeros(num_classes, dtype=np.float64),
    }
    for class_id in range(num_classes):
        pred_c = pred == class_id
        target_c = target == class_id
        totals["intersection"][class_id] = (pred_c & target_c).sum().item()
        totals["pred_pixels"][class_id] = pred_c.sum().item()
        totals["target_pixels"][class_id] = target_c.sum().item()
        totals["union"][class_id] = (pred_c | target_c).sum().item()
    return totals


def finish_metrics(totals: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    dice = (2 * totals["intersection"] + 1e-6) / (totals["pred_pixels"] + totals["target_pixels"] + 1e-6)
    iou = (totals["intersection"] + 1e-6) / (totals["union"] + 1e-6)
    recall = totals["intersection"] / np.maximum(totals["target_pixels"], 1)
    return {
        **totals,
        "dice": dice,
        "iou": iou,
        "recall": recall,
        "macro_dice_all": float(dice.mean()),
        "macro_dice_foreground": float(dice[1:].mean()),
    }


def evaluate_model_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fossa_threshold: float | None = None,
) -> dict[str, np.ndarray | float]:
    num_classes = len(CLASS_NAMES)
    totals = {
        "intersection": np.zeros(num_classes, dtype=np.float64),
        "pred_pixels": np.zeros(num_classes, dtype=np.float64),
        "target_pixels": np.zeros(num_classes, dtype=np.float64),
        "union": np.zeros(num_classes, dtype=np.float64),
    }
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            pred = prediction_from_logits(logits, fossa_threshold)
            batch_metrics = metrics_from_prediction(pred, masks, num_classes)
            for key in totals:
                totals[key] += batch_metrics[key]
    return finish_metrics(totals)


def grid_search_fossa_threshold(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[dict]]:
    rows = []
    best_threshold = FOSSA_THRESHOLDS[0]
    best_dice = -1.0
    for threshold in FOSSA_THRESHOLDS:
        metrics = evaluate_model_threshold(model, loader, device, threshold)
        fossa_dice = float(metrics["dice"][FOSSA_CLASS_ID])
        rows.append(
            {
                "threshold": threshold,
                "dice_background": float(metrics["dice"][0]),
                "dice_condyle": float(metrics["dice"][1]),
                "dice_fossa": fossa_dice,
                "macro_fg": float(metrics["macro_dice_foreground"]),
            }
        )
        if fossa_dice > best_dice:
            best_dice = fossa_dice
            best_threshold = threshold
    return best_threshold, rows


def evaluate_fixed_fossa_threshold(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> tuple[float, list[dict]]:
    metrics = evaluate_model_threshold(model, loader, device, threshold)
    row = {
        "threshold": threshold,
        "dice_background": float(metrics["dice"][0]),
        "dice_condyle": float(metrics["dice"][1]),
        "dice_fossa": float(metrics["dice"][FOSSA_CLASS_ID]),
        "macro_fg": float(metrics["macro_dice_foreground"]),
    }
    return threshold, [row]


def write_threshold_grid(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "dice_background", "dice_condyle", "dice_fossa", "macro_fg"],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    boundary_loss: nn.Module,
    args: argparse.Namespace,
) -> torch.Tensor:
    return (
        ce_loss(logits, masks)
        + args.dice_loss_weight * dice_loss(logits, masks)
        + args.boundary_loss_weight * boundary_loss(logits, masks)
    )


def train_one_fold(
    args: argparse.Namespace,
    fold: int,
    train_samples: list[Sample],
    valid_samples: list[Sample],
    test_samples: list[Sample],
    device: torch.device,
) -> dict:
    fold_dir = args.output_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_split(fold_dir / "split.csv", train_samples, valid_samples, test_samples)

    train_loader = make_loader(train_samples, args, train=True)
    valid_loader = make_loader(valid_samples, args, train=False)
    test_loader = make_loader(test_samples, args, train=False)

    model = UNetPlusPlus(num_classes=len(CLASS_NAMES), base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.05, 1.0, 4.0], dtype=torch.float32, device=device))
    dice_loss = DiceLoss(num_classes=len(CLASS_NAMES)).to(device)
    boundary_loss = BoundaryLoss(num_classes=len(CLASS_NAMES)).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = fold_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "valid_loss",
                "best_fossa_threshold",
                "dice_background",
                "dice_condyle",
                "dice_fossa",
                "mean_fg_dice",
            ]
        )

    best_valid_fossa_dice = -1.0
    best_fossa_threshold = FOSSA_THRESHOLDS[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = compute_loss(logits, masks, ce_loss, dice_loss, boundary_loss, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)
        train_loss /= max(train_count, 1)

        model.eval()
        valid_loss = 0.0
        valid_count = 0
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = compute_loss(logits, masks, ce_loss, dice_loss, boundary_loss, args)
                valid_loss += loss.item() * images.size(0)
                valid_count += images.size(0)
        valid_loss /= max(valid_count, 1)

        if args.fixed_fossa_threshold is None:
            epoch_threshold, threshold_rows = grid_search_fossa_threshold(model, valid_loader, device)
        else:
            epoch_threshold, threshold_rows = evaluate_fixed_fossa_threshold(
                model, valid_loader, device, args.fixed_fossa_threshold
            )
        valid_metrics = evaluate_model_threshold(model, valid_loader, device, epoch_threshold)
        dice = valid_metrics["dice"]
        fg_dice = float(valid_metrics["macro_dice_foreground"])
        fossa_dice = float(dice[FOSSA_CLASS_ID])

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, valid_loss, epoch_threshold, dice[0], dice[1], dice[2], fg_dice])

        print(
            f"fold {fold:02d} epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"best_thr_fossa={epoch_threshold:.2f} dice_bg={dice[0]:.4f} "
            f"dice_condyle={dice[1]:.4f} dice_fossa={dice[2]:.4f} mean_fg_dice={fg_dice:.4f}",
            flush=True,
        )

        if fossa_dice > best_valid_fossa_dice:
            best_valid_fossa_dice = fossa_dice
            best_fossa_threshold = epoch_threshold
            write_threshold_grid(fold_dir / "valid_fossa_threshold_grid.csv", threshold_rows)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "UNet++",
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                    "fold": fold,
                    "preprocess": "anatomic_pre_crop",
                    "fossa_threshold_grid": [args.fixed_fossa_threshold]
                    if args.fixed_fossa_threshold is not None
                    else FOSSA_THRESHOLDS,
                    "best_fossa_threshold": best_fossa_threshold,
                    "best_valid_fossa_dice": best_valid_fossa_dice,
                    "loss": "CrossEntropy + DiceLoss + BoundaryLoss",
                },
                fold_dir / "best_tmj_unetpp_boundary_fossa_threshold.pt",
            )

    checkpoint = torch.load(fold_dir / "best_tmj_unetpp_boundary_fossa_threshold.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_fossa_threshold = float(checkpoint["best_fossa_threshold"])
    valid_metrics = evaluate_model_threshold(model, valid_loader, device, best_fossa_threshold)
    test_metrics = evaluate_model_threshold(model, test_loader, device, best_fossa_threshold)
    write_metrics_csv(fold_dir / "valid_metrics_thresholded.csv", valid_metrics)
    write_metrics_csv(fold_dir / "test_metrics_thresholded.csv", test_metrics)
    return {
        "fold": fold,
        "best_valid_fossa_dice": float(checkpoint["best_valid_fossa_dice"]),
        "best_fossa_threshold": best_fossa_threshold,
        "valid": valid_metrics,
        "test": test_metrics,
    }


def write_cv_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "fold",
        "best_fossa_threshold",
        "best_valid_fossa_dice",
        "valid_dice_background",
        "valid_dice_condyle",
        "valid_dice_fossa",
        "valid_macro_fg",
        "test_dice_background",
        "test_dice_condyle",
        "test_dice_fossa",
        "test_macro_fg",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_from_existing_or_seeded(args: argparse.Namespace, samples: list[Sample]):
    samples_by_image = {str(sample.image_path): sample for sample in samples}
    if args.split_source_dir is not None and args.split_source_dir.exists():
        split_sets = []
        for fold in range(1, args.folds + 1):
            split_path = args.split_source_dir / f"fold_{fold:02d}" / "split.csv"
            if not split_path.exists():
                raise RuntimeError(f"Missing split file: {split_path}")
            split_sets.append((fold, *read_split(split_path, samples_by_image)))
        return split_sets

    labels = np.array([sample.severity for sample in samples])
    indices = np.arange(len(samples))
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    test_samples = [samples[i] for i in test_idx]
    trainval_labels = labels[trainval_idx]
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    split_sets = []
    for fold, (train_local, valid_local) in enumerate(cv.split(trainval_idx, trainval_labels), start=1):
        train_samples = [samples[i] for i in trainval_idx[train_local]]
        valid_samples = [samples[i] for i in trainval_idx[valid_local]]
        split_sets.append((fold, train_samples, valid_samples, test_samples))
    return split_sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet++ CV with Boundary Loss and fossa softmax thresholding.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\unet_runs\unetpp_cv3_boundary_fossa_threshold099"))
    parser.add_argument("--split-source-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--line-thickness", type=int, default=11)
    parser.add_argument("--crop-x-fraction", type=float, default=0.38)
    parser.add_argument("--crop-y-fraction", type=float, default=0.58)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--fixed-fossa-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No annotated samples found in {args.data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []
    for fold, train_samples, valid_samples, test_samples in split_from_existing_or_seeded(args, samples):
        result = train_one_fold(args, fold, train_samples, valid_samples, test_samples, device)
        valid_metrics = result["valid"]
        test_metrics = result["test"]
        summary_rows.append(
            {
                "fold": fold,
                "best_fossa_threshold": result["best_fossa_threshold"],
                "best_valid_fossa_dice": result["best_valid_fossa_dice"],
                "valid_dice_background": valid_metrics["dice"][0],
                "valid_dice_condyle": valid_metrics["dice"][1],
                "valid_dice_fossa": valid_metrics["dice"][2],
                "valid_macro_fg": valid_metrics["macro_dice_foreground"],
                "test_dice_background": test_metrics["dice"][0],
                "test_dice_condyle": test_metrics["dice"][1],
                "test_dice_fossa": test_metrics["dice"][2],
                "test_macro_fg": test_metrics["macro_dice_foreground"],
            }
        )
        write_cv_summary(args.output_dir / "cv_summary.csv", summary_rows)

    print(f"Saved CV summary: {args.output_dir / 'cv_summary.csv'}")
    for key in ["test_dice_background", "test_dice_condyle", "test_dice_fossa", "test_macro_fg"]:
        values = np.array([float(row[key]) for row in summary_rows])
        print(f"{key}: mean={values.mean():.4f} std={values.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
