# Best Models - TMJ 3-class Classification

This folder contains the latest validated non-leaky setup.

## Final selected model

The selected model is a simple probability blender:

`0.50 * CNN + 0.50 * RF requested clinical`

Selection criterion: best OOF CV=5 accuracy on the strict image-clinical matched cohort.

## Performance on matched cohort

- Matched cases: 150
- Labels: Mild, Normal, Severe
- Accuracy: 0.7200
- Balanced accuracy: 0.7087
- Macro-F1: 0.7039

## Folder structure

- `cnn_best/`: best CNN ensemble artifacts from `trial_020`, including base fold weights.
- `rf_requested_clinical/`: RF trained only with requested clinical variables, no diagnosis-derived features.
- `blender_50_50/`: final blend configuration, OOF probability files, and reproducible blend script.
- `reports/`: summaries, label checks, excluded cases, and performance graph.

## Leakage control

Excluded from the RF:

- `diagnosis`, `diagnosis1`, `diagnosis2`, `diagnosis3`
- all `diag_*` columns
- `condyle`

Included in the RF:

- demographics
- pain intensity and pain areas
- joint noises and tenderness
- mouth opening / movement measures
- mandibular deviation / dislocation
- dental history
- grouped medical history

## Reproduce the final OOF blend

```powershell
python best_models\blender_50_50\apply_blender_50_50.py `
  --cnn-probs best_models\blender_50_50\cnn_probs_image_matched_only.csv `
  --rf-probs best_models\rf_requested_clinical\rf_cv5_oof_predictions.csv `
  --output best_models\blender_50_50\final_50_50_oof_predictions.csv
```
