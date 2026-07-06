# Classification code

This folder contains classification, interpretability, occlusion and counterfactual code.

The current reference classifier is ResNet14 trained on segmentation-mask input.

## Input convention

The CNN classifiers expect:

```text
3 x 160 x 160 one-hot mask
channel 0 = background
channel 1 = condyle
channel 2 = fossa
```

Current PNG masks in `TMJ_clas/pred_fold02_fossa_erosion_top2_largest/masks/` may be visually encoded as:

```text
0   = background
76  = condyle
164 = fossa
```

When using these PNG masks, remap `0/76/164` to `0/1/2` before one-hot encoding.

## Training and blending scripts

| Script | Purpose |
|---|---|
| `clas_blend_optuna_cv3_Best_MODEL.py` | CNN/ResNet blend search with Optuna and meta-learner |
| `train_individual_resnets_cv3_oof.py` | Train individual ResNet variants with OOF predictions |
| `cnn_indiv_condyle_fossa_blender.py` | Separate condyle/fossa CNN branches followed by meta learner |
| `retrain_best_cnn_on_gt_seg_masks.py` | Re-evaluate best CNN using ground-truth segmentation masks |
| `evaluate_best_blend_holdout.py` | Holdout evaluation for blend experiments |

## Current ResNet14 analyses

| Script | Purpose |
|---|---|
| `counterfactual_morphology_resnet14.py` | ResNet14-only constrained morphology counterfactuals |
| `occlusion_local_mean_resnet14.py` | Remove condyle/fossa/both using local-mean replacement |
| `summarize_resnet14_occlusion_general_tests.py` | Global stats after occluding structures for all cases |
| `generate_gradcam_resnet14_multilayer_diagnostic.py` | Multi-layer Grad-CAM on the mask input |
| `resnet14_embedding_heads_cv.py` | Extract layer2/layer3 embeddings and train small heads |
| `make_resnet14_embedding_pca.py` | PCA visualisation of layer2 and layer3 embeddings |

## Reporting and figures

| Script | Purpose |
|---|---|
| `export_requested_morphology_ops_pdf_resnet14_readable.py` | Readable ResNet14 counterfactual PDF |
| `make_resnet14_counterfactual_global_occurrence_figures.py` | Global occurrence figures and heatmaps |
| `make_occlusion_mask_example_panel.py` | Visual check of mask variants |
| `make_occlusion_opg_overlay_example.py` | OPG overlay for occlusion variants |
| `make_local_mean_replacement_explanation.py` | Explains how local-mean replacement fills pixels |
| `export_resnet14_interpretability_pdf.py` | Combined ResNet14 interpretability report |

## Grad-CAM scripts

| Script | Purpose |
|---|---|
| `generate_cnn_best_gradcam_all_cases.py` | Shared CNN model definitions and Grad-CAM utilities |
| `generate_gradcam_all_meta_models_errors.py` | Grad-CAM for all meta-learner models on error cases |
| `generate_gradcam_all_meta_models_correct.py` | Grad-CAM for correct cases |
| `generate_gradcam_individual_resnets_cv3_multilayer_diagnostic.py` | Multi-layer Grad-CAM for individual ResNets |

## DINOv2 and ViT

| Folder | Purpose |
|---|---|
| `dinov2_tmd_experiment/` | DINOv2 CV3 experiment and attention visualisation |
| `vit_tmd_experiment/` | ViT-only CV3 experiment and attention visualisation |

## Reproduce important outputs

### ResNet14 counterfactuals

```powershell
python .\code_classification\counterfactual_morphology_resnet14.py
python .\code_classification\export_requested_morphology_ops_pdf_resnet14_readable.py
python .\code_classification\make_resnet14_counterfactual_global_occurrence_figures.py
```

### ResNet14 local-mean occlusions

```powershell
python .\code_classification\occlusion_local_mean_resnet14.py
python .\code_classification\summarize_resnet14_occlusion_general_tests.py
python .\code_classification\make_local_mean_replacement_explanation.py
```

### ResNet14 Grad-CAM

```powershell
python .\code_classification\generate_gradcam_resnet14_multilayer_diagnostic.py
```

### ResNet14 embeddings and PCA

```powershell
python .\code_classification\resnet14_embedding_heads_cv.py
python .\code_classification\make_resnet14_embedding_pca.py
```

## Interpretation notes

- Grad-CAM must be interpreted on the actual CNN input: the `160 x 160` segmentation mask.
- Full OPG overlays are anatomical references only.
- ResNet14 is the current selected CNN reference.
- DINOv2 CV3 currently uses full fine-tuning, not frozen linear probing.
