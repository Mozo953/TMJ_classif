from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["normal", "mild", "severe"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    case_id: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def case_id_from_path(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem.strip()


def collect_samples(data_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dossier de classe absent : {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(path.resolve(), label, class_name, case_id_from_path(path)))
    if not samples:
        raise RuntimeError(f"Aucune image trouvee dans {data_dir}")
    duplicates = sorted({s.case_id for s in samples if sum(t.case_id == s.case_id for t in samples) > 1})
    if duplicates:
        raise RuntimeError(f"Case IDs dupliques entre les classes : {duplicates}")
    return samples


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Conserve le ratio, centre l'image et complete jusqu'a size x size."""
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    fill = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    # Augmentations volontairement faibles pour ne pas inventer d'anatomie.
    fill = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.5:
        image = image.rotate(rng.uniform(-4.0, 4.0), resample=Image.Resampling.BILINEAR, fillcolor=fill)
    if rng.random() < 0.4:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class OPGDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool, seed: int) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = letterbox(image, self.image_size)
            if self.train:
                # Le worker PyTorch recoit sa propre graine ; random suffit ici.
                image = augment(image, random)
            tensor = image_to_tensor(image)
        return tensor, torch.tensor(sample.label, dtype=torch.long), index


def make_model(model_name: str, image_size: int, pretrained: bool, num_classes: int) -> nn.Module:
    # DINOv2 est ici utilise comme backbone ViT pur pre-entraine auto-supervise.
    # La Conv2d de patch_embed est seulement la projection lineaire standard
    # des patches, pas un backbone CNN.
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=image_size,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )


def make_loader(samples: list[Sample], args: argparse.Namespace, train: bool, fold: int) -> DataLoader:
    dataset = OPGDataset(samples, args.image_size, train, args.seed + fold)
    generator = torch.Generator().manual_seed(args.seed + fold)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def class_weights(samples: list[Sample], device: torch.device) -> torch.Tensor:
    counts = np.bincount([s.label for s in samples], minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = len(samples) / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        losses.append(criterion(logits, targets).item() * len(targets))
        labels.extend(targets.cpu().tolist())
        probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(labels), np.asarray(probabilities), float(sum(losses) / max(len(labels), 1))


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    y_pred = probabilities.argmax(axis=1)
    recalls = recall_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_per_class": {name: float(value) for name, value in zip(CLASS_NAMES, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))).tolist(),
    }


def save_confusion_matrix(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Classe predite")
    ax.set_ylabel("Classe reelle")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_fold(
    fold: int,
    train_samples: list[Sample],
    valid_samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    fold_dir = args.output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_samples, args, True, fold)
    valid_loader = make_loader(valid_samples, args, False, fold)
    model = make_model(args.model, args.image_size, True, len(CLASS_NAMES)).to(device)
    weights = class_weights(train_samples, device) if args.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    with (fold_dir / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "case_id", "class", "image"])
        for split, subset in (("train", train_samples), ("valid", valid_samples)):
            for sample in subset:
                writer.writerow([split, sample.case_id, sample.class_name, sample.path])

    best_score = -math.inf
    history: list[dict[str, float | int]] = []
    checkpoint_path = fold_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(targets)
            seen += len(targets)
        scheduler.step()

        y_true, probabilities, valid_loss = predict(model, valid_loader, device)
        metrics = metric_dict(y_true, probabilities)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "valid_loss": valid_loss,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "severe_recall": metrics["recall_per_class"]["severe"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"fold={fold} epoch={epoch:02d}/{args.epochs} "
            f"loss={row['train_loss']:.4f}/{valid_loss:.4f} "
            f"bal_acc={row['balanced_accuracy']:.3f} macro_f1={row['macro_f1']:.3f} "
            f"recall_severe={row['severe_recall']:.3f}",
            flush=True,
        )
        # Macro-F1 pilote le choix ; severe recall departage les ex-aequo.
        score = float(metrics["macro_f1"]) + 1e-4 * float(metrics["recall_per_class"]["severe"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "image_size": args.image_size,
                    "class_names": CLASS_NAMES,
                    "fold": fold,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                checkpoint_path,
            )

    with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    y_true, probabilities, _ = predict(model, valid_loader, device)
    metrics = metric_dict(y_true, probabilities)
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(np.asarray(metrics["confusion_matrix"]), fold_dir / "confusion_matrix.png", f"Fold {fold}")
    return probabilities, metrics


def write_oof(samples: list[Sample], folds: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    predictions = probabilities.argmax(axis=1)
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "image", "fold", "true_index", "true_class", "pred_index", "pred_class", "prob_normal", "prob_mild", "prob_severe"])
        for sample, fold, pred, probs in zip(samples, folds, predictions, probabilities):
            writer.writerow([sample.case_id, sample.path, fold, sample.label, sample.class_name, pred, CLASS_NAMES[pred], *probs.tolist()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification TMD par DINOv2 avec validation croisee et predictions OOF")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "TMJ_clas")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs_dinov2_cv3")
    parser.add_argument("--model", default="vit_small_patch14_dinov2")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    samples = collect_samples(args.data_dir)
    labels = np.asarray([s.label for s in samples])
    print(f"Images={len(samples)} classes={dict(zip(CLASS_NAMES, np.bincount(labels, minlength=3).tolist()))} device={device}")
    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probabilities = np.full((len(samples), len(CLASS_NAMES)), np.nan, dtype=np.float32)
    oof_folds = np.full(len(samples), -1, dtype=np.int64)
    fold_summaries: list[dict[str, object]] = []
    dummy_x = np.zeros(len(samples))
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(dummy_x, labels), start=1):
        train_samples = [samples[index] for index in train_indices]
        valid_samples = [samples[index] for index in valid_indices]
        probabilities, metrics = train_fold(fold, train_samples, valid_samples, args, device)
        oof_probabilities[valid_indices] = probabilities
        oof_folds[valid_indices] = fold
        fold_summaries.append({"fold": fold, **metrics})

    if np.isnan(oof_probabilities).any() or (oof_folds < 0).any():
        raise RuntimeError("Predictions OOF incompletes")
    oof_metrics = metric_dict(labels, oof_probabilities)
    write_oof(samples, oof_folds, oof_probabilities, args.output_dir)
    (args.output_dir / "oof_metrics.json").write_text(json.dumps(oof_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "fold_metrics.json").write_text(json.dumps(fold_summaries, indent=2), encoding="utf-8")
    matrix = np.asarray(oof_metrics["confusion_matrix"])
    save_confusion_matrix(matrix, args.output_dir / "oof_confusion_matrix.png", "Matrice de confusion OOF")
    with (args.output_dir / "oof_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row.tolist()])
    print(json.dumps(oof_metrics, indent=2))


if __name__ == "__main__":
    main()
