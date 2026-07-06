# Calibration metrics

This folder contains Expected Calibration Error and Brier score results for the CNN/base/blender models.

## Definitions

### Mean confidence

For each image:

```text
confidence = max(P(Mild), P(Normal), P(Severe))
```

`mean_confidence` is the average of this value over all 150 images.

### ECE

Expected Calibration Error compares predicted confidence with empirical accuracy across probability bins.

Lower is better.

### Brier score

Multiclass Brier score:

```text
mean over samples of sum_k (p_k - one_hot_k)^2
```

Lower is better.

## Results

| Model | Accuracy | ECE 10 bins | ECE 15 bins | Brier multiclass | Mean confidence |
|---|---:|---:|---:|---:|---:|
| ResNet14 | 0.6867 | 0.1693 | 0.1693 | 0.4701 | 0.5173 |
| ResNet20 | 0.6533 | 0.1072 | 0.1179 | 0.4549 | 0.5573 |
| ResNet26 | 0.6733 | 0.1259 | 0.1515 | 0.4417 | 0.5683 |
| Small CNN | 0.6800 | 0.1349 | 0.1349 | 0.4412 | 0.5451 |
| Depthwise CNN | 0.6600 | 0.1959 | 0.1959 | 0.5312 | 0.4662 |
| Blender RF | 0.6333 | 0.0507 | 0.0556 | 0.4156 | 0.6771 |

## Files

| File | Role |
|---|---|
| `ece_brier_summary.csv` | Main calibration table |
| `ece_10bin_details.csv` | Per-bin details for ECE |
| `calibration_definition.json` | Formula and source file metadata |
