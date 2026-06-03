# Matched CNN + Requested Clinical RF CV=5

## Filtrage strict

- Clinical source: `outputs/clinical_pipeline/clinical_merged_with_diag_features.csv`
- CNN source: `TMJ_clas/clas_runs/blend_resnets_optuna_cv5_verbose/trial_020/meta_oof_predictions.csv`
- Cases conserves: 150 cases ayant a la fois une image/CNN et une ligne clinique.
- Cases cliniques exclus: 50, car sans probabilites CNN associees.
- Images/CNN sans clinical case: 0.
- Ground truth utilisee: colonne clinique `severity`.
- Incoherence reperee: `Case ID 35`, clinical=`Severe`, CNN label=`Mild`.

## RF utilise

Le RF est relance avec `--feature-set requested_clinical`.

Variables incluses:

- demographics: `age`, `age_visit`, `age_range`, `gender`, `race`
- pain/joint signs: `pain_scor`, `painarea_*`, `ttpp_*`, `crep`, `click`
- functional assessment: `dev`, `disl`, `mmo`, `mio`, `lat_R`, `lat_L`, `maxprot`
- dental/medical history: `teeth_wea`, `teeth_grin`, `hard_food`, `MD_*`

Variables exclues pour eviter le leakage:

- `diagnosis`, `diagnosis1`, `diagnosis2`, `diagnosis3`
- toutes les features `diag_*`
- `condyle`

## Resultats OOF CV=5

| Modele | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| CNN seul | 0.6733 | 0.6607 | 0.6527 |
| RF requested clinical seul | 0.6933 | 0.6865 | 0.6825 |
| Meta-blender logistique CNN+RF | 0.6600 | 0.6579 | 0.6591 |
| Meilleur blend pondere simple | 0.7200 | 0.7087 | 0.7039 |

Meilleur blend pondere simple:

- poids CNN: 0.50
- poids RF requested clinical: 0.50

## Matrice de confusion RF requested clinical

| True / Pred | Mild | Normal | Severe |
|---|---:|---:|---:|
| Mild | 35 | 2 | 19 |
| Normal | 1 | 49 | 0 |
| Severe | 22 | 2 | 20 |

## Matrice de confusion meta-blender

| True / Pred | Mild | Normal | Severe |
|---|---:|---:|---:|
| Mild | 31 | 2 | 23 |
| Normal | 4 | 46 | 0 |
| Severe | 21 | 1 | 22 |

## Conclusion

L'experience correcte, sans features de diagnostic, donne des performances plus realistes. Le RF requested clinical seul est meilleur que le CNN seul. Le meta-blender logistique n'apporte pas de gain, mais un blend simple 50/50 CNN + RF ameliore legerement le score global.

Le meilleur compromis actuel sans leakage est donc le blend pondere simple:

`0.50 * CNN + 0.50 * RF requested clinical`

