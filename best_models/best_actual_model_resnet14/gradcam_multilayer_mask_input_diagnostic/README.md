# ResNet14 Grad-CAM diagnostics

This folder contains Grad-CAM analyses for the current ResNet14 model.

## Important interpretation rule

ResNet14 does not see the raw OPG.

It sees:

```text
3 x 160 x 160 one-hot segmentation mask
```

Therefore:

```text
Primary interpretation = Grad-CAM on the 160 x 160 mask input.
Full OPG overlays = anatomical references only.
```

## Layers analysed

| Layer name | Spatial size | Meaning |
|---|---:|---|
| `layer1` | 160 x 160 | early spatial features |
| `layer2` | 80 x 80 | intermediate features |
| `layer3` | 40 x 40 | deepest convolutional features |

## Main outputs

| File/folder | Role |
|---|---|
| `figures_multilayer_mask_first/` | Per-case figures, mask-first interpretation |
| `heatmaps_npy/` | Raw CAM arrays |
| `gradcam_multilayer_diagnostics.csv` | Gradient/CAM statistics per case and layer |
| `contact_sheet_multilayer_errors_only.jpg` | Error-case contact sheet |
| `contact_sheet_multilayer_mild.jpg` | True Mild contact sheet |
| `contact_sheet_multilayer_normal.jpg` | True Normal contact sheet |
| `contact_sheet_multilayer_severe.jpg` | True Severe contact sheet |
| `run_config.json` | Run configuration |

## Reproduce

```powershell
python .\code_classification\generate_gradcam_resnet14_multilayer_diagnostic.py
```
