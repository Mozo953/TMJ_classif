from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from clas_cv_best import (
    CLASS_NAMES,
    SimpleResNet14,
    collect_mask_samples,
    evaluate,
    make_loader,
    seed_everything,
    write_confusion,
    write_predictions,
)


def write_split(path: Path, train_samples, valid_samples, test_samples) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "mask"])
        for split, samples in [("train", train_samples), ("valid", valid_samples), ("test", test_samples)]:
            for sample in samples:
                writer.writerow([split, sample.label_name, str(sample.mask_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ResNet14 classifier with a single train/valid/test split.")
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_fold02_fossa_erosion_top2_meanprob\masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\resnet14_top2_meanprob_single_split"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--valid-size", type=float, default=0.15)
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
    indices = np.arange(len(samples))

    trainvalid_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    trainvalid_labels = labels[trainvalid_idx]
    relative_valid_size = args.valid_size / (1.0 - args.test_size)
    train_idx, valid_idx = train_test_split(
        trainvalid_idx,
        test_size=relative_valid_size,
        random_state=args.seed,
        stratify=trainvalid_labels,
    )

    train_samples = [samples[i] for i in train_idx]
    valid_samples = [samples[i] for i in valid_idx]
    test_samples = [samples[i] for i in test_idx]
    write_split(args.output_dir / "split.csv", train_samples, valid_samples, test_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_samples, args, train=True)
    valid_loader = make_loader(valid_samples, args, train=False)
    test_loader = make_loader(test_samples, args, train=False)

    model = SimpleResNet14(base_channels=args.base_channels, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history_rows = []
    best_valid_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels_batch)
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
            f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} acc={valid_metrics['accuracy']:.4f} "
            f"macro_f1={valid_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = valid_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "SimpleResNet14",
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                    "best_valid_macro_f1": best_valid_macro_f1,
                    "mask_root": str(args.mask_root),
                },
                args.output_dir / "best_resnet14_mask_classifier.pt",
            )

    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(args.output_dir / "best_resnet14_mask_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    valid_metrics = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    write_predictions(args.output_dir / "valid_predictions.csv", valid_samples, valid_metrics)
    write_predictions(args.output_dir / "test_predictions.csv", test_samples, test_metrics)
    write_confusion(args.output_dir / "valid_confusion_matrix.csv", valid_metrics["confusion"])
    write_confusion(args.output_dir / "test_confusion_matrix.csv", test_metrics["confusion"])

    summary = []
    for split, metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        row = {"split": split}
        for key in ["loss", "accuracy", "macro_f1", "macro_precision", "macro_recall"]:
            row[key] = metrics[key]
        summary.append(row)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "loss", "accuracy", "macro_f1", "macro_precision", "macro_recall"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved best model: {args.output_dir / 'best_resnet14_mask_classifier.pt'}")
    print(f"Saved summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
