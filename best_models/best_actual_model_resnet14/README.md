# Best actual model - ResNet14

This folder contains the current selected CNN model:

```text
ResNet14 alone
```

The model is trained on segmentation masks, not raw panoramic radiographs.

## Input

```text
3 x 160 x 160 one-hot mask
channel 0 = background
channel 1 = condyle
channel 2 = fossa
```

Current display masks may use:

```text
0   = background
76  = condyle
164 = fossa
```

These values must be remapped to `0 / 1 / 2` before one-hot encoding.

## Checkpoints

| File | Role |
|---|---|
| `fold_01.pt` ... `fold_05.pt` | ResNet14 fold checkpoints |
| `BEST_ACTUAL_MODEL.json` | Confirms current selected model |
| `source_all_models_metrics.csv` | Source comparison metrics |

## Main metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.6867 |
| Balanced accuracy | 0.6544 |
| Macro F1 | 0.6255 |
| Recall Mild | 0.8772 |
| Recall Normal | 0.9000 |
| Recall Severe | 0.1860 |

## Calibration

Calibration was computed in:

```text
best_models/cnn_best_retrain_check/calibration_metrics/
```

| Metric | Value |
|---|---:|
| ECE 10 bins | 0.1693 |
| ECE 15 bins | 0.1693 |
| Brier multiclass | 0.4701 |
| Brier Mild | 0.1979 |
| Brier Normal | 0.0931 |
| Brier Severe | 0.1791 |
| Mean confidence | 0.5173 |

## Subfolders

| Folder | Purpose |
|---|---|
| `counterfactual_morphology_all_cases/` | Constrained morphological counterfactuals |
| `occlusion_replace_structures_local_mean/` | Local-mean occlusion experiments |
| `gradcam_multilayer_mask_input_diagnostic/` | Multi-layer Grad-CAM diagnostics |
| `embedding_heads_cv/` | Layer2/layer3 embeddings, small heads, PCA |

## Interpretation rule

Because the model sees masks, not raw OPG images:

```text
Interpret Grad-CAM and counterfactuals on the 160 x 160 mask input.
Full OPG overlays are anatomical references only.
```
