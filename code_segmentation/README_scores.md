# Segmentation - scores cles

## Meilleur modele

Le meilleur pipeline retenu est `seg_cv_best.py` avec inference via `infer_cv2_with_fossa_erosion_grid.py`.
Il correspond au modele UNet++ CV fold 02 avec post-traitement fossa erosion et seuil fossa principalement a `0.95`.

## Scores principaux

- Validation CV, meilleur fold: fold 02, fossa Dice `0.5374`, macro foreground `0.6482`.
- Test CV moyen sur 5 folds: macro foreground Dice `0.6304`.
- Test CV moyen: condyle Dice `0.7406`, glenoid fossa Dice `0.5201`.
- Evaluation prediction top2 mean probability: macro foreground Dice `0.6232`, macro all Dice `0.7464`.

## Organisation

- Code principal: `seg_cv_best.py`, `infer_cv2_with_fossa_erosion_grid.py`.
- Experimentation: anciens entrainements, essais UNet/UNet++, calculs Dice, overlays et evaluation du modele OPGSeg.
