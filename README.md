# TMJ / TMD classification project

This workspace contains the experimental pipeline for temporo-mandibular disorder classification from panoramic dental radiographs and segmentation-derived masks.

The project has three main blocks:

1. segmentation of TMJ structures;
2. classification into `Normal`, `Mild`, `Severe`;
3. interpretability and robustness analyses of the selected CNN model.

## Current reference model

The current selected CNN reference is:

```text
best actual model = ResNet14 alone
```

Artifacts are stored in:

```text
best_models/best_actual_model_resnet14/
```

Important: ResNet14 is trained on segmentation masks, not on raw OPG pixels.

## CNN input

```text
one-hot segmentation mask, 3 x 160 x 160
channel 0 = background
channel 1 = condyle
channel 2 = fossa
```

Some current PNG masks are visually encoded as:

```text
0   = background
76  = condyle
164 = fossa
```

Scripts must remap these values to `0 / 1 / 2` before one-hot encoding.

## Main folders

| Folder | Role |
|---|---|
| `TMJ_clas/` | Classification images and segmentation masks |
| `TMJ_Xrays/` | Raw OPG image copy |
| `code_segmentation/` | Segmentation training, inference, Dice evaluation and overlays |
| `code_classification/` | Classification, interpretability, occlusion, Grad-CAM, counterfactuals |
| `best_models/` | Validated outputs, selected models, metrics and reports |
| `outputs/` / `output/` | Additional reports and generated artifacts |

## Core result locations

| Analysis | Output folder |
|---|---|
| Current ResNet14 model | `best_models/best_actual_model_resnet14/` |
| Counterfactual morphology | `best_models/best_actual_model_resnet14/counterfactual_morphology_all_cases/` |
| Local-mean occlusions | `best_models/best_actual_model_resnet14/occlusion_replace_structures_local_mean/` |
| Grad-CAM diagnostics | `best_models/best_actual_model_resnet14/gradcam_multilayer_mask_input_diagnostic/` |
| Embedding heads and PCA | `best_models/best_actual_model_resnet14/embedding_heads_cv/` |
| CNN calibration | `best_models/cnn_best_retrain_check/calibration_metrics/` |
| DINOv2 CV3 experiment | `code_classification/dinov2_tmd_experiment/outputs_dinov2_cv3/` |

## Key metrics

### ResNet14

| Metric | Value |
|---|---:|
| Accuracy | 0.6867 |
| Balanced accuracy | 0.6544 |
| Macro F1 | 0.6255 |
| Recall Mild | 0.8772 |
| Recall Normal | 0.9000 |
| Recall Severe | 0.1860 |
| ECE 10 bins | 0.1693 |
| Brier multiclass | 0.4701 |

### DINOv2 full fine-tuning

| Metric | Value |
|---|---:|
| Accuracy | 0.7467 |
| Balanced accuracy | 0.7389 |
| Macro F1 | 0.7407 |
| Recall Normal | 0.9800 |
| Recall Mild | 0.7018 |
| Recall Severe | 0.5349 |

The current DINOv2 experiment is full fine-tuning, not frozen linear probing.

## Reading order

1. `best_models/best_actual_model_resnet14/README.md`
2. `code_classification/README.md`
3. `code_segmentation/README.md`
4. `best_models/best_actual_model_resnet14/counterfactual_morphology_all_cases/README.md`
5. `best_models/best_actual_model_resnet14/occlusion_replace_structures_local_mean/README.md`
6. `best_models/best_actual_model_resnet14/gradcam_multilayer_mask_input_diagnostic/README.md`

## Safety note

Many scripts use absolute paths under:

```text
C:\Users\sadmin\Desktop\mozo
```

For this reason, files were documented in place instead of being moved. Moving scripts without updating hard-coded paths may break reproducibility.
