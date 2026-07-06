# DINOv2 TMD experiment results

This folder contains the DINOv2 experiment for 3-class TMD classification.

## Important training detail

The current DINOv2 experiment is full fine-tuning.

The script uses:

```python
optimizer = torch.optim.AdamW(model.parameters(), ...)
```

and does not set `requires_grad=False` for the backbone.

Therefore:

```text
DINOv2 backbone + classification head are trained.
This is not frozen linear probing.
```

## Configuration

| Parameter | Value |
|---|---|
| Model | `vit_small_patch14_dinov2` |
| Image size | 512 |
| Folds | 3 |
| Epochs | 30 |
| Batch size | 8 |
| Learning rate | 3e-5 |
| Weight decay | 0.05 |
| Label smoothing | 0.05 |
| Class-weighted loss | true |
| Seed | 42 |

## OOF results

| Metric | Value |
|---|---:|
| Accuracy | 0.7467 |
| Balanced accuracy | 0.7389 |
| Macro F1 | 0.7407 |
| Recall Normal | 0.9800 |
| Recall Mild | 0.7018 |
| Recall Severe | 0.5349 |

## Confusion matrix

Order: `Normal / Mild / Severe`

| True \ Pred | Normal | Mild | Severe |
|---|---:|---:|---:|
| Normal | 49 | 1 | 0 |
| Mild | 0 | 40 | 17 |
| Severe | 0 | 20 | 23 |

## Files

| File/folder | Role |
|---|---|
| `outputs_dinov2_cv3/oof_metrics.json` | OOF metrics |
| `outputs_dinov2_cv3/oof_confusion_matrix.csv` | OOF confusion matrix |
| `outputs_dinov2_cv3/oof_predictions.csv` | OOF predictions |
| `outputs_dinov2_cv3/fold_metrics.json` | Per-fold metrics |
| `outputs_dinov2_cv3/attention_*` | Attention visualisations |

## Suggested next experiment

Because the dataset is small, the next cleaner comparison should test:

1. frozen DINOv2 + Logistic Regression;
2. frozen DINOv2 + linear layer;
3. frozen DINOv2 + light MLP;
4. partial fine-tuning of the last transformer block only.
