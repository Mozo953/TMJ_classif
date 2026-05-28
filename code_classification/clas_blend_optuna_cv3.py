from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["mild", "normal", "severe"]
MASK_VALUES = [0, 1, 2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BASE_MODEL_NAMES = ["resnet14", "resnet20", "resnet26", "small_cnn", "depthwise_cnn"]


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


def collect_samples(mask_root: Path) -> list[MaskSample]:
    samples = []
    for label_idx, label_name in enumerate(CLASS_NAMES):
        class_dir = mask_root / label_name
        if not class_dir.exists():
            raise RuntimeError(f"Missing class folder: {class_dir}")
        for mask_path in sorted(class_dir.iterdir()):
            if mask_path.name.startswith("._") or mask_path.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append(MaskSample(mask_path=mask_path, label=label_idx, label_name=label_name))
    if not samples:
        raise RuntimeError(f"No masks found in {mask_root}")
    return samples


def mask_to_one_hot(mask: np.ndarray) -> np.ndarray:
    return np.stack([(mask == value).astype(np.float32) for value in MASK_VALUES], axis=0)


class MaskDataset(Dataset):
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
        return torch.from_numpy(mask_to_one_hot(mask)), torch.tensor(sample.label, dtype=torch.long)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class MaskResNet(nn.Module):
    def __init__(self, blocks: list[int], base_channels: int, dropout: float, num_classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.layer1 = self._make_layer(base_channels, channels[0], blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[2], num_classes)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x).flatten(1))


class SmallCNN(nn.Module):
    def __init__(self, base_channels: int, dropout: float, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 5, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


class DepthwiseCNN(nn.Module):
    def __init__(self, base_channels: int, dropout: float, num_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, base_channels, 1)
        self.blocks = nn.Sequential(
            self._dw_block(base_channels, base_channels * 2, stride=2),
            self._dw_block(base_channels * 2, base_channels * 4, stride=2),
            self._dw_block(base_channels * 4, base_channels * 4, stride=1),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Linear(base_channels * 4, num_classes)

    @staticmethod
    def _dw_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(x), inplace=True)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def make_model(name: str, base_channels: int, dropout: float) -> nn.Module:
    if name == "resnet14":
        return MaskResNet([2, 2, 2], base_channels, dropout)
    if name == "resnet20":
        return MaskResNet([3, 3, 3], base_channels, dropout)
    if name == "resnet26":
        return MaskResNet([4, 4, 4], base_channels, dropout)
    if name == "small_cnn":
        return SmallCNN(base_channels, dropout)
    if name == "depthwise_cnn":
        return DepthwiseCNN(base_channels, dropout)
    raise ValueError(f"Unknown model: {name}")


def make_loader(samples: list[MaskSample], image_size: int, batch_size: int, train: bool, num_workers: int) -> DataLoader:
    ds = MaskDataset(samples, image_size=image_size, train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def train_base_model(
    model_name: str,
    train_samples: list[MaskSample],
    valid_samples: list[MaskSample],
    args: argparse.Namespace,
    params: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> np.ndarray:
    model = make_model(model_name, params["base_channels"], params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = make_loader(train_samples, params["image_size"], params["batch_size"], True, args.num_workers)
    valid_loader = make_loader(valid_samples, params["image_size"], params["batch_size"], False, args.num_workers)

    best_f1 = -1.0
    best_state = None
    for _epoch in range(1, params["epochs"] + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probs, y_true = predict_proba(model, valid_loader, device)
        y_pred = probs.argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "model_state": model.state_dict(), "params": params, "class_names": CLASS_NAMES}, checkpoint_path)
    probs, _y_true = predict_proba(model, valid_loader, device)
    return probs


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(y_all, axis=0)


def train_oof_base_models(
    samples: list[MaskSample],
    labels: np.ndarray,
    args: argparse.Namespace,
    params: dict,
    trial_dir: Path,
    device: torch.device,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_by_model = []
    for model_name in BASE_MODEL_NAMES:
        oof = np.zeros((len(samples), len(CLASS_NAMES)), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(np.arange(len(samples)), labels), start=1):
            train_samples = [samples[i] for i in train_idx]
            valid_samples = [samples[i] for i in valid_idx]
            ckpt = trial_dir / "base_models" / model_name / f"fold_{fold:02d}.pt"
            oof[valid_idx] = train_base_model(model_name, train_samples, valid_samples, args, params, device, ckpt)
        np.save(trial_dir / f"oof_{model_name}.npy", oof)
        oof_by_model.append(oof)
    return np.concatenate(oof_by_model, axis=1)


def fit_meta_model(trial: optuna.Trial, x_oof: np.ndarray, y: np.ndarray, seed: int) -> tuple[object, dict, float, np.ndarray]:
    meta_type = trial.suggest_categorical("meta_type", ["logistic", "random_forest", "extra_trees"])
    scaler = None
    if meta_type == "logistic":
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_oof)
        model = LogisticRegression(
            C=trial.suggest_float("meta_logreg_c", 0.01, 20.0, log=True),
            max_iter=2000,
            class_weight=trial.suggest_categorical("meta_logreg_class_weight", [None, "balanced"]),
            random_state=seed,
            multi_class="auto",
        )
    elif meta_type == "random_forest":
        x_fit = x_oof
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("meta_rf_estimators", 80, 300),
            max_depth=trial.suggest_int("meta_rf_max_depth", 2, 8),
            min_samples_leaf=trial.suggest_int("meta_rf_min_leaf", 1, 5),
            class_weight=trial.suggest_categorical("meta_rf_class_weight", [None, "balanced"]),
            random_state=seed,
        )
    else:
        x_fit = x_oof
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int("meta_et_estimators", 80, 300),
            max_depth=trial.suggest_int("meta_et_max_depth", 2, 8),
            min_samples_leaf=trial.suggest_int("meta_et_min_leaf", 1, 5),
            class_weight=trial.suggest_categorical("meta_et_class_weight", [None, "balanced"]),
            random_state=seed,
        )

    # Meta CV avoids scoring the meta learner on exactly the same rows it fits.
    meta_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    meta_oof = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float32)
    for train_idx, valid_idx in meta_cv.split(x_oof, y):
        if scaler is not None:
            fold_scaler = StandardScaler()
            x_train = fold_scaler.fit_transform(x_oof[train_idx])
            x_valid = fold_scaler.transform(x_oof[valid_idx])
        else:
            x_train = x_oof[train_idx]
            x_valid = x_oof[valid_idx]
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(x_train, y[train_idx])
        meta_oof[valid_idx] = fold_model.predict_proba(x_valid)
    score = f1_score(y, meta_oof.argmax(axis=1), average="macro", zero_division=0)

    model.fit(x_fit, y)
    wrapped = {"model": model, "scaler": scaler, "meta_type": meta_type}
    metrics = classification_metrics(y, meta_oof.argmax(axis=1))
    return wrapped, metrics, score, meta_oof


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_predictions(path: Path, samples: list[MaskSample], y_true: np.ndarray, probs: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mask", "true_label", "pred_label", "prob_mild", "prob_normal", "prob_severe"])
        for sample, true_idx, pred_idx, prob in zip(samples, y_true, probs.argmax(axis=1), probs):
            writer.writerow([str(sample.mask_path), CLASS_NAMES[int(true_idx)], CLASS_NAMES[int(pred_idx)], *prob.tolist()])


def objective(trial: optuna.Trial, samples: list[MaskSample], labels: np.ndarray, args: argparse.Namespace, device: torch.device) -> float:
    params = {
        "epochs": trial.suggest_int("epochs", args.min_epochs, args.max_epochs),
        "image_size": trial.suggest_categorical("image_size", args.image_sizes),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
        "base_channels": trial.suggest_categorical("base_channels", args.base_channels_choices),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.35),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.12),
    }
    trial_dir = args.output_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / "base_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    x_oof = train_oof_base_models(samples, labels, args, params, trial_dir, device)
    np.save(trial_dir / "oof_blend_features.npy", x_oof)
    meta, metrics, score, meta_oof = fit_meta_model(trial, x_oof, labels, args.seed)
    joblib.dump(meta, trial_dir / "meta_model.joblib")
    np.save(trial_dir / "meta_oof_probs.npy", meta_oof)
    write_predictions(trial_dir / "meta_oof_predictions.csv", samples, labels, meta_oof)

    with (trial_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"score_macro_f1": score, **metrics}, f, indent=2)
    with (trial_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, confusion_matrix(labels, meta_oof.argmax(axis=1), labels=[0, 1, 2])):
            writer.writerow([name, *row.tolist()])

    trial.set_user_attr("trial_dir", str(trial_dir))
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return score


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna CV3 blender: 3 ResNets + 2 CNNs + adaptive meta learner.")
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_fold02_fossa_erosion_top2_largest\masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\blend_resnets_optuna_cv3_top2_largest"),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--image-sizes", type=parse_int_list, default=[160, 224])
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[4, 8])
    parser.add_argument("--base-channels-choices", type=parse_int_list, default=[16, 24, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--study-name", default="blend_resnets_optuna_cv3")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
        args.trials = 1
        args.min_epochs = 1
        args.max_epochs = 1
        args.image_sizes = [64]
        args.batch_sizes = [8]
        args.base_channels_choices = [8]

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args.mask_root)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with (args.output_dir / "dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mask", "label"])
        for sample in samples:
            writer.writerow([sample.mask_path, sample.label_name])

    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{(args.output_dir / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(lambda trial: objective(trial, samples, labels, args, device), n_trials=args.trials)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(args.output_dir / "optuna_trials.csv", index=False)
    best = study.best_trial
    with (args.output_dir / "best_trial.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "number": best.number,
                "value_macro_f1": best.value,
                "params": best.params,
                "user_attrs": best.user_attrs,
            },
            f,
            indent=2,
        )
    print(f"Best trial: {best.number}")
    print(f"Best macro_f1: {best.value:.4f}")
    print(f"Best trial dir: {best.user_attrs.get('trial_dir')}")
    print(f"Saved study: {args.output_dir}")


if __name__ == "__main__":
    main()
