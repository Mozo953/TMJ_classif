from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
logging.getLogger("albumentations.check_version").setLevel(logging.CRITICAL)

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["background", "condyle", "glenoid_fossa"]
CONDYLE_LABELS = {"R_condyle", "L_condyle", "R_condoyle", "L_condoyle"}
FOSSA_LABELS = {"R_glenoid_fossa", "L_glenoid_fossa", "R_glennoidfossa", "L_glennoidfossa"}
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
DATA_FOLDERS = {"mild", "normal", "severe"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    json_path: Path
    severity: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_for_json(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists() and not candidate.name.startswith("._"):
            return candidate
    return None


def collect_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for folder in sorted([p for p in root.iterdir() if p.is_dir() and p.name in DATA_FOLDERS]):
        for json_path in sorted(folder.glob("*.json")):
            if json_path.name.startswith("._"):
                continue
            image_path = find_image_for_json(json_path)
            if image_path is not None:
                samples.append(Sample(image_path=image_path, json_path=json_path, severity=folder.name))
    return samples


def polygon_points(shape: dict) -> np.ndarray:
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.round(points).astype(np.int32)


def json_to_mask(json_path: Path, height: int, width: int, line_thickness: int) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label in CONDYLE_LABELS:
            class_id = 1
        elif label in FOSSA_LABELS:
            class_id = 2
        else:
            continue

        pts = polygon_points(shape)
        if pts.size == 0:
            continue

        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "linestrip" or pts.shape[0] < 3:
            cv2.polylines(mask, [pts], isClosed=False, color=class_id, thickness=line_thickness)
        else:
            cv2.fillPoly(mask, [pts], color=class_id)
    return mask


def crop_boxes(width: int, height: int, x_fraction: float, y_fraction: float) -> list[tuple[int, int, int, int]]:
    crop_w = max(1, min(int(round(width * x_fraction)), width))
    crop_h = max(1, min(int(round(height * y_fraction)), height))
    return [(0, 0, crop_w, crop_h), (width - crop_w, 0, width, crop_h)]


def apply_pre_crop(image: np.ndarray, mask: np.ndarray, x_fraction: float, y_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    image_out = np.zeros_like(image)
    mask_out = np.zeros_like(mask)
    for x1, y1, x2, y2 in crop_boxes(width, height, x_fraction, y_fraction):
        image_out[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        mask_out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return image_out, mask_out


def train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Rotate(limit=(5, 15), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.Rotate(limit=(-15, -5), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                ],
                p=0.7,
            ),
            A.RandomScale(scale_limit=(-0.15, 0.15), interpolation=cv2.INTER_LINEAR, p=0.4),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomCrop(image_size, image_size),
            A.ElasticTransform(alpha=25, sigma=6, alpha_affine=4, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3),
            A.GridDistortion(p=0.3),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.35),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.25, p=0.4),
            A.GaussNoise(var_limit=(5.0, 35.0), p=0.3),
        ]
    )


def valid_transform(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA)])


class TMJPreCropDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        line_thickness: int,
        x_fraction: float,
        y_fraction: float,
        transform: A.Compose,
    ):
        self.samples = samples
        self.image_size = image_size
        self.line_thickness = line_thickness
        self.x_fraction = x_fraction
        self.y_fraction = y_fraction
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        mask = json_to_mask(sample.json_path, image.shape[0], image.shape[1], self.line_thickness)
        image, mask = apply_pre_crop(image, mask, self.x_fraction, self.y_fraction)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"].astype(np.float32) / 255.0
        mask = augmented["mask"].astype(np.int64)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(mask)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base: int = 32):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.conv0_0 = ConvBlock(in_channels, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4])
        self.conv0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.conv1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.conv2_1 = ConvBlock(nb[2] + nb[3], nb[2])
        self.conv3_1 = ConvBlock(nb[3] + nb[4], nb[3])
        self.conv0_2 = ConvBlock(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = ConvBlock(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = ConvBlock(nb[2] * 2 + nb[3], nb[2])
        self.conv0_3 = ConvBlock(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = ConvBlock(nb[1] * 3 + nb[2], nb[1])
        self.conv0_4 = ConvBlock(nb[0] * 4 + nb[1], nb[0])
        self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    @staticmethod
    def upsample(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, size=like.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0, x0_0)], dim=1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1, x0_0)], dim=1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2, x0_0)], dim=1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3, x0_0)], dim=1))
        return self.final(x0_4)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh = torch.nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        start = 1 if self.ignore_background else 0
        probs = probs[:, start:]
        target_oh = target_oh[:, start:]
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - dice.mean()


def class_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, np.ndarray]:
    pred = torch.argmax(logits, dim=1)
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


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray | float]:
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
            batch_metrics = class_metrics_from_logits(logits, masks, num_classes)
            for key in totals:
                totals[key] += batch_metrics[key]
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


def write_metrics_csv(path: Path, metrics: dict[str, np.ndarray | float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "dice", "iou", "pixel_recall", "target_pixels", "pred_pixels"])
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name, metrics["dice"][i], metrics["iou"][i], metrics["recall"][i], int(metrics["target_pixels"][i]), int(metrics["pred_pixels"][i])])
        writer.writerow(["macro_all", metrics["macro_dice_all"], "", "", "", ""])
        writer.writerow(["macro_foreground", metrics["macro_dice_foreground"], "", "", "", ""])


def write_split(path: Path, train_samples: list[Sample], valid_samples: list[Sample], test_samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "severity", "image", "json"])
        for split, split_samples in [("train", train_samples), ("valid", valid_samples), ("test", test_samples)]:
            for sample in split_samples:
                writer.writerow([split, sample.severity, str(sample.image_path), str(sample.json_path)])


def make_loader(samples: list[Sample], args: argparse.Namespace, train: bool) -> DataLoader:
    transform = train_transform(args.image_size) if train else valid_transform(args.image_size)
    ds = TMJPreCropDataset(samples, args.image_size, args.line_thickness, args.crop_x_fraction, args.crop_y_fraction, transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)


def train_one_fold(args: argparse.Namespace, fold: int, train_samples: list[Sample], valid_samples: list[Sample], test_samples: list[Sample], device: torch.device) -> dict:
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
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_path = fold_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "dice_background", "dice_condyle", "dice_fossa", "mean_fg_dice"])

    best_fg_dice = -1.0
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
                loss = ce_loss(logits, masks) + args.dice_loss_weight * dice_loss(logits, masks)
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
                loss = ce_loss(logits, masks) + args.dice_loss_weight * dice_loss(logits, masks)
                valid_loss += loss.item() * images.size(0)
                valid_count += images.size(0)
        valid_loss /= max(valid_count, 1)
        valid_metrics = evaluate_model(model, valid_loader, device)
        dice = valid_metrics["dice"]
        fg_dice = float(valid_metrics["macro_dice_foreground"])

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, valid_loss, dice[0], dice[1], dice[2], fg_dice])

        print(
            f"fold {fold:02d} epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"dice_bg={dice[0]:.4f} dice_condyle={dice[1]:.4f} "
            f"dice_fossa={dice[2]:.4f} mean_fg_dice={fg_dice:.4f}",
            flush=True,
        )

        if fg_dice > best_fg_dice:
            best_fg_dice = fg_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "UNet++",
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                    "fold": fold,
                    "preprocess": "anatomic_pre_crop",
                    "best_valid_foreground_dice": best_fg_dice,
                },
                fold_dir / "best_tmj_unetpp_precrop.pt",
            )

    checkpoint = torch.load(fold_dir / "best_tmj_unetpp_precrop.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    valid_metrics = evaluate_model(model, valid_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    write_metrics_csv(fold_dir / "valid_metrics.csv", valid_metrics)
    write_metrics_csv(fold_dir / "test_metrics.csv", test_metrics)
    return {"fold": fold, "best_valid_fg_dice": best_fg_dice, "valid": valid_metrics, "test": test_metrics}


def write_cv_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "fold",
        "best_valid_fg_dice",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet++ 5-fold CV with anatomic pre-crop inputs.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\unet_runs\unetpp_cv5_precrop_e50"))
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No annotated samples found in {args.data_root}")
    labels = np.array([sample.severity for sample in samples])
    indices = np.arange(len(samples))
    trainval_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=labels)
    test_samples = [samples[i] for i in test_idx]
    trainval_labels = labels[trainval_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    summary_rows = []
    for fold, (train_local, valid_local) in enumerate(cv.split(trainval_idx, trainval_labels), start=1):
        train_samples = [samples[i] for i in trainval_idx[train_local]]
        valid_samples = [samples[i] for i in trainval_idx[valid_local]]
        result = train_one_fold(args, fold, train_samples, valid_samples, test_samples, device)
        valid_metrics = result["valid"]
        test_metrics = result["test"]
        summary_rows.append(
            {
                "fold": fold,
                "best_valid_fg_dice": result["best_valid_fg_dice"],
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
