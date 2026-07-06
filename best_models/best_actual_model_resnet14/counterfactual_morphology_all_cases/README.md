# ResNet14 counterfactual morphology

This folder contains constrained morphological counterfactual analyses for ResNet14.

## Goal

For each case, apply anatomically plausible mask modifications and measure whether the model prediction changes.

The analysis asks:

```text
Which morphological change is sufficient to move the prediction toward another class?
```

## Main outputs

| File/folder | Role |
|---|---|
| `all_counterfactual_results.csv` | All generated perturbations and probability changes |
| `summary.json` | Global run summary |
| `summary_by_case.csv` | Per-case counterfactual summary |
| `summary_by_side.csv` | Left/right/both side summary |
| `summary_by_transformation.csv` | Summary by transformation |
| `summary_by_transformation_and_true_class.csv` | Summary by transformation and true label |
| `requested_morphology_operations_tables/` | Readable PDFs and CSV tables |
| `global_occurrence_figures_readable/` | Global occurrence plots and heatmaps |

## Main operations

| Operation | Meaning |
|---|---|
| `condyle_erosion` | progressively reduce condyle size |
| `condyle_dilation` | enlarge condyle |
| `condyle_fossa_distance_increase` | increase condyle-fossa space |
| `condyle_fossa_distance_decrease` | decrease condyle-fossa space |
| `condyle_translation` | move one condyle |
| `condyle_partial_removal` | remove part of condyle |
| `condyle_local_irregularity` | local pixel-level irregularity |
| fossa transforms | perturb fossa geometry |

## Current global counts

| Quantity | Value |
|---|---:|
| Cases | 150 |
| Generated rows | 31412 |
| Valid perturbations | 30957 |
| Rejected perturbations | 455 |
| Any prediction flips | 3359 |
| Flips to Severe | 505 |

## Readable PDF

Most readable PDF:

```text
requested_morphology_operations_tables/TABLE_REQUESTED_MORPHOLOGY_OPERATIONS_EN_RESNET14_READABLE.pdf
```

## Occurrence figures

Clean global plots:

```text
global_occurrence_figures_readable/
```

| File | Role |
|---|---|
| `00_global_occurrence_dashboard.png` | Global counts, flips, destinations and confidence |
| `02_heatmap_flip_rate.png` | Flip-rate heatmap by class and operation |
| `02_heatmap_delta_severe.png` | Mean change in P(Severe) |

## Reproduce

```powershell
python .\code_classification\counterfactual_morphology_resnet14.py
python .\code_classification\export_requested_morphology_ops_pdf_resnet14_readable.py
python .\code_classification\make_resnet14_counterfactual_global_occurrence_figures.py
```
