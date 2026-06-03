# Clinical preprocessing + RF + adaptive CNN/RF blender

This folder contains the clean end-to-end clinical workflow.

For the full interpretation of the experiments, read:

`EXPERIMENT_SUMMARY.md`

## 1. Full clinical preprocessing and Random Forest

Runs:

1. `normal_clinical(1).xlsx` raw text preprocessing.
2. `clinicaldataa.csv` alignment.
3. Normal + Mild + Severe merge.
4. Diagnosis normalization and `diag_*` binary features.
5. Random Forest 3-class CV=5 with seed `42`.
6. Optuna light search with `5` trials per fold.

```powershell
python run_preprocess_and_rf.py
```

Outputs:

- `../outputs/clinical_pipeline/clinical_merged_with_diag_features.csv`
- `../outputs/rf_3class_cv5_optuna/rf_cv5_oof_predictions.csv`
- `../outputs/rf_3class_cv5_optuna/rf_3class_final_model.joblib`
- `../outputs/rf_3class_cv5_optuna/rf_final_feature_importances.csv`

To train the RF without diagnosis-derived `diag_*` features:

```powershell
python ../train_rf_3class_cv5_optuna.py --exclude-diagnosis-features --output-dir ../outputs/rf_3class_cv5_optuna_no_diag
```

## 2. Adaptive blender without CNN inference

This script does not run image inference. It reuses precomputed CNN probabilities
and RF probabilities.

Default CNN source:

`../../TMJ_clas/clas_runs/blend_resnets_optuna_cv5_verbose/trial_020/meta_oof_predictions.csv`

Default RF source:

`../outputs/rf_3class_cv5_optuna/rf_cv5_oof_predictions.csv`

```powershell
python adaptive_cnn_rf_blender_no_inference.py
```

To force CNN weight to `0.6`:

```powershell
python adaptive_cnn_rf_blender_no_inference.py --no-tune-weights --min-cnn-weight 0.6 --max-cnn-weight 0.6 --output-dir ../outputs/adaptive_cnn_rf_blender_cnn06
```

Outputs:

- `../outputs/adaptive_cnn_rf_blender/blender_predictions.csv`
- `../outputs/adaptive_cnn_rf_blender/blender_metrics.json`
- `../outputs/adaptive_cnn_rf_blender/blender_confusion_matrix.csv`

## Important

For rows with no CNN probabilities, the blender automatically falls back to RF
only. This is expected for Normal cases if no normal CNN OOF predictions exist.
The script calibrates the adaptive CNN weight bounds by default on available
OOF predictions; use `--no-tune-weights` to force fixed weights.
