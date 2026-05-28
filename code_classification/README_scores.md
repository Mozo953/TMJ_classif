# Classification - scores cles

## Meilleur modele

Le meilleur pipeline retenu est `clas_blend_optuna_cv3.py`, un blend de plusieurs CNN/ResNet avec meta-modele.
Le meilleur essai Optuna est le trial 011 avec meta-modele logistic.

## Scores principaux

- CV Optuna trial 011: accuracy `0.6933`, macro-F1 `0.6927`, macro precision `0.6930`, macro recall `0.6929`.
- Meilleur modele de base dans le trial 011: ResNet20, accuracy `0.6933`, macro-F1 `0.7028`.
- Holdout du pipeline final: ResNet20 accuracy `0.6522`, macro-F1 `0.6444`.
- Holdout meta-logistic: accuracy `0.4783`, macro-F1 `0.4722`.

## Organisation

- Code principal: `clas_blend_optuna_cv3.py`, `evaluate_best_blend_holdout.py`, `summarize_blend_base_models.py`.
- Experimentation: entrainement simple split, CV precedent, scripts d'overlays et visualisations de probabilites.
