from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from clas_blend_optuna_cv3 import (
    BASE_MODEL_NAMES,
    CLASS_NAMES,
    collect_samples,
    make_loader,
    make_model,
    predict_proba,
    seed_everything,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, confusion_matrix(y_true, y_pred, labels=[0, 1, 2])):
            writer.writerow([name, *row.tolist()])


def write_predictions(path: Path, samples, y_true: np.ndarray, probs: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mask", "true_label", "pred_label", "prob_mild", "prob_normal", "prob_severe"])
        for sample, true_idx, pred_idx, prob in zip(samples, y_true, probs.argmax(axis=1), probs):
            writer.writerow([str(sample.mask_path), CLASS_NAMES[int(true_idx)], CLASS_NAMES[int(pred_idx)], *prob.tolist()])


def train_model_fixed(model_name: str, train_samples, params: dict, args: argparse.Namespace, device: torch.device, checkpoint_path: Path):
    model = make_model(model_name, params["base_channels"], params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = make_loader(train_samples, params["image_size"], params["batch_size"], True, args.num_workers)

    model.train()
    for epoch in range(1, params["epochs"] + 1):
        running = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            count += x.size(0)
        if epoch == 1 or epoch == params["epochs"] or epoch % max(params["epochs"] // 5, 1) == 0:
            print(f"{model_name} final epoch {epoch:03d}/{params['epochs']} loss={running / max(count, 1):.4f}", flush=True)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "model_state": model.state_dict(), "params": params, "class_names": CLASS_NAMES}, checkpoint_path)
    return model


def train_oof_model(model_name: str, trainval_samples, trainval_labels: np.ndarray, params: dict, args: argparse.Namespace, device: torch.device, out_dir: Path) -> np.ndarray:
    cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(trainval_samples), len(CLASS_NAMES)), dtype=np.float32)
    indices = np.arange(len(trainval_samples))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(indices, trainval_labels), start=1):
        train_samples = [trainval_samples[i] for i in train_idx]
        valid_samples = [trainval_samples[i] for i in valid_idx]
        model = make_model(model_name, params["base_channels"], params["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        train_loader = make_loader(train_samples, params["image_size"], params["batch_size"], True, args.num_workers)
        valid_loader = make_loader(valid_samples, params["image_size"], params["batch_size"], False, args.num_workers)

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

        probs, _ = predict_proba(model, valid_loader, device)
        oof[valid_idx] = probs
        (out_dir / model_name).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_name": model_name, "model_state": model.state_dict(), "params": params, "class_names": CLASS_NAMES},
            out_dir / model_name / f"oof_fold_{fold:02d}.pt",
        )
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best Optuna blender with an external hold-out test set.")
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_fold02_fossa_erosion_top2_largest\masks"),
    )
    parser.add_argument(
        "--best-trial",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\blend_resnets_optuna_cv3_top2_largest\best_trial.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\best_blend_holdout_top2_largest"),
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.best_trial.open("r", encoding="utf-8") as f:
        best = json.load(f)
    params = best["params"].copy()
    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        params["epochs"] = 1
        params["image_size"] = 64
        params["batch_size"] = 8
        params["base_channels"] = min(int(params["base_channels"]), 8)

    base_params = {
        "epochs": int(params["epochs"]),
        "image_size": int(params["image_size"]),
        "batch_size": int(params["batch_size"]),
        "base_channels": int(params["base_channels"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "dropout": float(params["dropout"]),
        "label_smoothing": float(params["label_smoothing"]),
    }

    samples = collect_samples(args.mask_root)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    idx = np.arange(len(samples))
    trainval_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=labels)
    trainval_samples = [samples[i] for i in trainval_idx]
    test_samples = [samples[i] for i in test_idx]
    trainval_labels = labels[trainval_idx]
    test_labels = labels[test_idx]

    with (args.output_dir / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "mask"])
        for sample in trainval_samples:
            writer.writerow(["trainval", sample.label_name, sample.mask_path])
        for sample in test_samples:
            writer.writerow(["test", sample.label_name, sample.mask_path])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_features = []
    test_features = []
    base_rows = []

    for model_name in BASE_MODEL_NAMES:
        print(f"OOF training {model_name}", flush=True)
        oof = train_oof_model(model_name, trainval_samples, trainval_labels, base_params, args, device, args.output_dir / "oof_models")
        oof_features.append(oof)

        print(f"Final trainval training {model_name}", flush=True)
        model = train_model_fixed(model_name, trainval_samples, base_params, args, device, args.output_dir / "final_base_models" / f"{model_name}.pt")
        test_loader = make_loader(test_samples, base_params["image_size"], base_params["batch_size"], False, args.num_workers)
        probs, _ = predict_proba(model, test_loader, device)
        test_features.append(probs)
        pred = probs.argmax(axis=1)
        row = {"model": model_name, **metrics(test_labels, pred)}
        base_rows.append(row)
        write_predictions(args.output_dir / f"test_predictions_{model_name}.csv", test_samples, test_labels, probs)
        write_confusion(args.output_dir / f"test_confusion_{model_name}.csv", test_labels, pred)

    x_oof = np.concatenate(oof_features, axis=1)
    x_test = np.concatenate(test_features, axis=1)
    np.save(args.output_dir / "trainval_oof_features.npy", x_oof)
    np.save(args.output_dir / "test_blend_features.npy", x_test)

    scaler = StandardScaler()
    x_oof_scaled = scaler.fit_transform(x_oof)
    x_test_scaled = scaler.transform(x_test)
    meta = LogisticRegression(
        C=float(params.get("meta_logreg_c", 1.0)),
        class_weight=params.get("meta_logreg_class_weight", None),
        max_iter=2000,
        random_state=args.seed,
    )
    meta.fit(x_oof_scaled, trainval_labels)
    meta_probs = meta.predict_proba(x_test_scaled)
    meta_pred = meta_probs.argmax(axis=1)
    meta_metrics = {"model": "meta_logistic", **metrics(test_labels, meta_pred)}

    joblib.dump({"model": meta, "scaler": scaler, "params": params}, args.output_dir / "meta_model_holdout.joblib")
    write_predictions(args.output_dir / "test_predictions_meta.csv", test_samples, test_labels, meta_probs)
    write_confusion(args.output_dir / "test_confusion_meta.csv", test_labels, meta_pred)

    all_rows = base_rows + [meta_metrics]
    with (args.output_dir / "holdout_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"base_params": base_params, "meta_params": params, "best_trial": str(args.best_trial)}, f, indent=2)

    print("Hold-out metrics:")
    for row in all_rows:
        print(f"{row['model']}: acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} precision={row['macro_precision']:.4f} recall={row['macro_recall']:.4f}")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
