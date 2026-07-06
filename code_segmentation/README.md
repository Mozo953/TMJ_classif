# Segmentation code

This folder contains segmentation training, inference, Dice evaluation and overlay scripts.

The segmentation target structures used by the classifier are:

```text
background
condyle
glenoid fossa / fossa
```

## Main scripts

| Script | Purpose |
|---|---|
| `seg_cv_best.py` | Main UNet++ CV training script |
| `seg_cv_unet_same_method.py` | UNet trained with the same method as UNet++ |
| `infer_cv2_with_fossa_erosion_grid.py` | Inference and fossa erosion/post-processing grid |
| `compute_cv5_oof_dice_per_case.py` | Dice evaluation per case |
| `make_label_vs_prediction_overlays.py` | Label/prediction overlays |
| `make_labelme_segmentation_overlays.py` | LabelMe segmentation visual checks |
| `create_labelme_segmentation_dataset.py` | Convert LabelMe annotations to dataset format |

## Current segmentation summary

| Model | Background Dice | Condyle Dice | Fossa Dice | Macro foreground Dice |
|---|---:|---:|---:|---:|
| UNet++ | 0.9926 | 0.7406 | 0.5201 | 0.6304 |
| UNet | 0.9932 | 0.7544 | 0.5069 | 0.6306 |

UNet and UNet++ are similar overall. UNet++ is slightly better for fossa, while UNet is slightly better for condyle in the reported run.

## Important output folders

| Folder | Purpose |
|---|---|
| `TMJ_clas/pred_fold02_fossa_erosion_top2_largest/masks/` | Main predicted masks used by the CNN classifier |
| `TMJ_clas/unet_runs/` | UNet/UNet++ training and evaluation runs |
| `TMJ_clas/best_seg/` | Best segmentation outputs and variants |

## Mask encoding note

Some PNG masks are stored for display as:

```text
0   = background
76  = condyle
164 = fossa
```

Classification scripts must remap these values to:

```text
0 = background
1 = condyle
2 = fossa
```

before one-hot encoding.
