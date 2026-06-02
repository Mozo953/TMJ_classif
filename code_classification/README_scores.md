# Classification - scores cles

## Meilleur modele

Le meilleur pipeline retenu est `clas_blend_optuna_cv3.py`, un blend de plusieurs CNN/ResNet avec meta-modele.
La derniere relance complete utilise `cv=5`; le meilleur essai Optuna est `trial_000` avec meta-modele logistic.

## Scores principaux

- CV=5 Optuna trial 000: accuracy `0.6933`, macro-F1 `0.6903`, macro precision `0.6894`, macro recall `0.6954`.
- Parametres du meilleur trial: `epochs=45`, `image_size=160`, `batch_size=4`, `base_channels=32`, meta-modele `logistic` avec `class_weight=balanced`.
- Holdout du pipeline final: ResNet20 accuracy `0.6522`, macro-F1 `0.6444`.
- Holdout meta-logistic: accuracy `0.4783`, macro-F1 `0.4722`.

## Organisation

- Code principal: `clas_blend_optuna_cv3.py`, `evaluate_best_blend_holdout.py`, `summarize_blend_base_models.py`.
- Experimentation: entrainement simple split, CV precedent, scripts d'overlays et visualisations de probabilites.
