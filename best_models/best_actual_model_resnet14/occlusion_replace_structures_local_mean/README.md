# ResNet14 local-mean occlusion

This folder contains global occlusion tests for ResNet14.

## Goal

Remove anatomical structures from every mask and observe how the prediction changes.

| Mode | Meaning |
|---|---|
| `baseline` | original mask |
| `localmean_condyle` | condyles replaced by local mean |
| `localmean_fossa` | fossae replaced by local mean |
| `localmean_both` | condyles and fossae replaced by local mean |

## Local-mean replacement

The replacement is not black fill and not hard background.

For selected pixels only:

```text
1. compute a local class-probability average in a 31 x 31 neighborhood;
2. replace the selected one-hot pixel by this local average;
3. renormalize the 3 channels.
```

Example:

```text
before: [background=0, condyle=1, fossa=0]
after:  [background=0.85, condyle=0.10, fossa=0.05]
```

## General results

| Test | Accuracy | Balanced accuracy | Macro F1 | Pred Mild | Pred Normal | Pred Severe | Changed vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.6867 | 0.6544 | 0.6255 | 87 | 51 | 12 | 0.0% |
| Remove condyles | 0.3467 | 0.3362 | 0.2856 | 93 | 2 | 55 | 50.7% |
| Remove fossae | 0.3400 | 0.3411 | 0.1832 | 0 | 148 | 2 | 65.3% |
| Remove condyles + fossae | 0.4267 | 0.4524 | 0.4255 | 30 | 25 | 95 | 65.3% |

## Main outputs

| File/folder | Role |
|---|---|
| `local_mean_occlusion_summary.csv` | Global counts by mode |
| `local_mean_occlusion_long.csv` | Long format, one row per case and mode |
| `local_mean_occlusion_wide.csv` | Wide format for case-level comparison |
| `general_tests_all_cases/` | General summary figures and transition matrices |
| `figures/` | Case-level visualisations |
| `contact_sheet_true_mild.jpg` | Contact sheet for true Mild cases |
| `contact_sheet_true_normal.jpg` | Contact sheet for true Normal cases |
| `contact_sheet_true_severe.jpg` | Contact sheet for true Severe cases |
| `example_masks_visual_check/` | Detailed visual explanation examples |

## Reproduce

```powershell
python .\code_classification\occlusion_local_mean_resnet14.py
python .\code_classification\summarize_resnet14_occlusion_general_tests.py
python .\code_classification\make_occlusion_mask_example_panel.py
python .\code_classification\make_occlusion_opg_overlay_example.py
python .\code_classification\make_local_mean_replacement_explanation.py
```
