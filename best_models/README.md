# Best models and validated outputs

This folder stores trained checkpoints, OOF predictions, reports and selected model artifacts.

## Current reference

The current CNN reference model is:

```text
best_actual_model_resnet14/
```

The user explicitly corrected the reference model to ResNet14. Older folders such as `best_actual_model_resnet20/` are kept for traceability but should not be treated as the current best CNN.

## Main folders

| Folder | Purpose |
|---|---|
| `best_actual_model_resnet14/` | Current selected CNN model and all ResNet14 interpretability analyses |
| `cnn_best_retrain_check/` | Retrained base CNNs, OOF probabilities, calibration metrics |
| `cnn_best/` | Earlier CNN ensemble/blender artifacts and interpretability reports |
| `cnn_indiv_condyle_fossa_blender/` | Condyle-only and fossa-only CNN branch experiments |
| `individual_resnets_cv3_oof/` | Individual ResNet20/ResNet26 CV3 experiments |
| `logreg_meta_3models/` | Logistic regression meta-learner combining model probabilities |
| `logreg_meta_3models_plus_clinical/` | Model probabilities plus tabular/clinical data |
| `logreg_dinov2_plus_clinical/` | DINOv2 plus clinical/tabular data |
| `rf_requested_clinical/` | Random forest using requested clinical variables |
| `reports/` | Additional comparison reports and error analyses |

## Calibration results

Calibration files are located at:

```text
best_models/cnn_best_retrain_check/calibration_metrics/
```

| Model | Accuracy | ECE 10 bins | Brier multiclass |
|---|---:|---:|---:|
| ResNet14 | 0.6867 | 0.1693 | 0.4701 |
| ResNet20 | 0.6533 | 0.1072 | 0.4549 |
| ResNet26 | 0.6733 | 0.1259 | 0.4417 |
| Small CNN | 0.6800 | 0.1349 | 0.4412 |
| Depthwise CNN | 0.6600 | 0.1959 | 0.5312 |
| Blender RF | 0.6333 | 0.0507 | 0.4156 |

## Historical note

There are older experiments combining CNN probabilities with Random Forest clinical models. Those are preserved, but the current CNN reference for interpretability and counterfactual analyses is ResNet14 alone.
