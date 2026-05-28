from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["mild", "normal", "severe"]
MASK_CLASS_VALUES = [0, 1, 2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskSample:
    mask_path: Path
    label: int
    label_name: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collect_mask_samples(mask_root: Path) -> list[MaskSample]:
    samples: list[MaskSample] = []
    for label_idx, label_name in enumerate(CLASS_NAMES):
        class_dir = mask_root / label_name
        if not class_dir.exists():
            raise RuntimeError(f"Missing class folder: {class_dir}")
        for mask_path in sorted(class_dir.iterdir()):
            if mask_path.name.startswith("._") or mask_path.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append(MaskSample(mask_path=mask_path, label=label_idx, label_name=label_name))
    if not samples:
        raise RuntimeError(f"No mask images found in {mask_root}")
    return samples


def mask_to_one_hot(mask: np.ndarray) -> np.ndarray:
    channels = [(mask == value).astype(np.float32) for value in MASK_CLASS_VALUES]
    return np.stack(channels, axis=0)


class MaskSeverityDataset(Dataset):
    def __init__(self, samples: list[MaskSample], image_size: int, train: bool) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        mask = np.array(Image.open(sample.mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = cv2.resize(mask.astype(np.uint8), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.train and random.random() < 0.5:
            mask = np.ascontiguousarray(mask[:, ::-1])

        x = mask_to_one_hot(mask)
        y = np.int64(sample.label)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SimpleResNet14(nn.Module):
    """Small ResNet for mask-only severity classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(base_channels, base_channels),
            BasicBlock(base_channels, base_channels),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(base_channels, base_channels * 2, stride=2),
            BasicBlock(base_channels * 2, base_channels * 2),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(base_channels * 2, base_channels * 4, stride=2),
            BasicBlock(base_channels * 4, base_channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def make_loader(samples: list[MaskSample], args: argparse.Namespace, train: bool) -> DataLoader:
    dataset = MaskSeverityDataset(samples, image_size=args.image_size, train=train)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    y_prob = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            losses.append(loss.item() * images.size(0))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    total = max(len(y_true), 1)
    return {
        "loss": float(np.sum(losses) / total),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def write_predictions(path: Path, samples: list[MaskSample], metrics: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mask", "true_label", "pred_label", "prob_mild", "prob_normal", "prob_severe"])
        for sample, pred, probs in zip(samples, metrics["y_pred"], metrics["y_prob"]):
            writer.writerow([str(sample.mask_path), sample.label_name, CLASS_NAMES[pred], *probs])


def write_confusion(path: Path, confusion: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for label_name, row in zip(CLASS_NAMES, confusion):
            writer.writerow([label_name, *row.tolist()])


def train_one_fold(
    args: argparse.Namespace,
    fold: int,
    train_samples: list[MaskSample],
    valid_samples: list[MaskSample],
    device: torch.device,
) -> dict:
    fold_dir = args.output_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(train_samples, args, train=True)
    valid_loader = make_loader(valid_samples, args, train=False)

    model = SimpleResNet14(base_channels=args.base_channels, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    with (fold_dir / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "mask"])
        for split, split_samples in [("train", train_samples), ("valid", valid_samples)]:
            for sample in split_samples:
                writer.writerow([split, sample.label_name, str(sample.mask_path)])

    best_macro_f1 = -1.0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)
        train_loss /= max(train_count, 1)

        valid_metrics = evaluate(model, valid_loader, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_macro_precision": valid_metrics["macro_precision"],
                "valid_macro_recall": valid_metrics["macro_recall"],
            }
        )
        print(
            f"fold {fold:02d} epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"acc={valid_metrics['accuracy']:.4f} macro_f1={valid_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "SimpleResNet14",
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                    "fold": fold,
                    "best_valid_macro_f1": best_macro_f1,
                },
                fold_dir / "best_resnet14_mask_classifier.pt",
            )

    with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(fold_dir / "best_resnet14_mask_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    valid_metrics = evaluate(model, valid_loader, device)
    write_predictions(fold_dir / "valid_predictions.csv", valid_samples, valid_metrics)
    write_confusion(fold_dir / "valid_confusion_matrix.csv", valid_metrics["confusion"])
    return {
        "fold": fold,
        "best_valid_macro_f1": checkpoint["best_valid_macro_f1"],
        "valid_accuracy": valid_metrics["accuracy"],
        "valid_macro_f1": valid_metrics["macro_f1"],
        "valid_macro_precision": valid_metrics["macro_precision"],
        "valid_macro_recall": valid_metrics["macro_recall"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify TMJ severity from segmentation masks with a small ResNet14.")
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\seg_best_unet++_cv2_inférence_threshold=0.95\masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\resnet14_masks_cv"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_mask_samples(args.mask_root)
    labels = np.array([sample.label for sample in samples])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows = []
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(np.arange(len(samples)), labels), start=1):
        train_samples = [samples[i] for i in train_idx]
        valid_samples = [samples[i] for i in valid_idx]
        summary_rows.append(train_one_fold(args, fold, train_samples, valid_samples, device))
        with (args.output_dir / "cv_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved CV summary: {args.output_dir / 'cv_summary.csv'}")
    for key in ["valid_accuracy", "valid_macro_f1", "valid_macro_precision", "valid_macro_recall"]:
        values = np.array([float(row[key]) for row in summary_rows])
        print(f"{key}: mean={values.mean():.4f} std={values.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
