# Synthese des transformations cliniques, Random Forest et blender CNN/RF

## Objectif

L'objectif etait de construire une pipeline complete et automatique capable de :

1. Lire les fichiers cliniques bruts et structures.
2. Transformer les notes cliniques en variables tabulaires exploitables.
3. Normaliser les diagnostics en categories medicales coherentes.
4. Fusionner les cas `Normal`, `Mild` et `Severe`.
5. Entrainer une Random Forest 3 classes.
6. Comparer les performances avec et sans features issues du diagnostic.
7. Tester un blender entre les probabilites CNN deja calculees et le modele clinique RF.

La pipeline finale ne relance pas la segmentation ni l'inference CNN. Elle reutilise les sorties deja disponibles.

## Organisation

Sous-dossier principal :

`TMD_DiagnosisTool/clinical_preprocessing_rf_blender/`

Scripts :

- `run_preprocess_and_rf.py` : lance le preprocessing complet puis la Random Forest.
- `adaptive_cnn_rf_blender_no_inference.py` : combine les probabilites CNN deja calculees avec les probabilites RF.
- `README.md` : commandes rapides.
- `EXPERIMENT_SUMMARY.md` : ce document de synthese.

Modules utilises par la pipeline :

- `TMD_DiagnosisTool/preprocess_clinical_features.py`
- `TMD_DiagnosisTool/normalize_diagnosis_features.py`
- `TMD_DiagnosisTool/run_full_clinical_pipeline.py`
- `TMD_DiagnosisTool/train_rf_3class_cv5_optuna.py`

## Donnees fusionnees

Deux sources cliniques sont utilisees :

- `normal_clinical(1).xlsx`, feuille `Control group`
- `clinicaldataa.csv`

Le fichier normal brut est transforme automatiquement par regles NLP medicales :

- extraction de `click`, `crep`, `mmo`, `mio`, `lat_R`, `lat_L`, `maxprot`
- extraction des signes lateraux `L`, `R`, `B`
- extraction des antecedents medicaux `MD_*`
- extraction des habitudes dentaires `teeth_grin`, `teeth_wea`, `hard_food`
- normalisation de `severity`
- decoupage des diagnostics `diagnosis`, `diagnosis1`, `diagnosis2`, `diagnosis3`

Le fichier TMD deja structure est aligne sur le meme schema.

Resultat fusionne :

```text
Total cas : 200
Mild      : 104
Severe    : 46
Normal    : 50
```

Fichiers produits :

- `TMD_DiagnosisTool/outputs/clinical_pipeline/clinical_merged_structured.csv`
- `TMD_DiagnosisTool/outputs/clinical_pipeline/clinical_merged_with_diag_features.csv`
- `TMD_DiagnosisTool/outputs/clinical_pipeline/clinical_merged_diagnosis_normalization_table.csv`

## Normalisation des diagnostics

Les colonnes analysees sont :

- `diagnosis`
- `diagnosis1`
- `diagnosis2`
- `diagnosis3`

Les variantes et fautes de frappe sont regroupees.

Exemples :

```text
Impacted tooh                  -> Impacted Tooth
Impacted tooth (#38 #48)       -> Impacted Tooth
Non-funtional tooth            -> Non-Functional Tooth
Buried tooth                   -> Buried Tooth
Generalised Severe Gingivitis  -> Gingivitis
TMJ Athralgia                  -> TMJ Arthralgia
TMJ synovitis/ capsulities     -> TMJ Synovitis/Capsulitis
Osterarthrosis                 -> TMJ Osteoarthritis/Osteoarthrosis
```

Les diagnostics normalises sont ensuite regroupes en familles :

- Tooth Eruption Disorders
- Dental Diseases
- Periodontal Diseases
- Functional Disorders
- Occlusal / Skeletal Disorders
- TMJ / Masticatory Disorders
- TMJ / Intra-Articular Disorders
- TMJ / Inflammatory / Pain Disorders
- TMJ / Degenerative Disorders
- TMJ / Condylar Pathology
- TMJ / Dislocation Disorders
- Parafunctional Habits

Des colonnes binaires `diag_*` sont creees pour le machine learning.

Exemples :

```text
diag_impacted_tooth
diag_nonfunctional_tooth
diag_disc_displacement_with_reduction
diag_tmj_condition
diag_tmj_osteoarthritis_osteoarthrosis
diag_tmj_synovitis_capsulitis
diag_masticatory_myalgia
```

## Protocole Random Forest

Le modele utilise une Random Forest 3 classes :

```text
Classes : Mild, Normal, Severe
CV      : StratifiedKFold, n_splits=5
Seed    : 42
Optuna  : 5 trials par fold
Metric  : macro-F1
```

Le seed `42` est le meme que celui utilise dans la pipeline blender de classification.

## Experience 1 : RF avec diagnostics normalises

Les features incluent les variables cliniques structurees et les colonnes `diag_*`.

Resultats OOF CV=5 :

```text
Accuracy          : 0.9250
Balanced accuracy : 0.9075
Macro-F1          : 0.9179
Macro precision   : 0.9328
Macro recall      : 0.9075
```

Par classe :

```text
Mild    F1 = 0.93 | recall = 0.96 | n = 104
Normal  F1 = 1.00 | recall = 1.00 | n = 50
Severe  F1 = 0.82 | recall = 0.76 | n = 46
```

Matrice de confusion :

```text
             pred_Mild  pred_Normal  pred_Severe
true_Mild        100        0            4
true_Normal        0       50            0
true_Severe       11        0           35
```

Features les plus importantes :

```text
diag_impacted_tooth
condyle_NA
age
age_visit
condyle_L
mmo_NA
diag_nonfunctional_tooth
diag_tmj_condition
diag_disc_displacement_with_reduction
```

Interpretation :

Le modele avec diagnostics est le plus performant. Les features `diag_*` apportent une information medicale utile, mais elles representent une information post-evaluation clinique. Cela est acceptable si le modele est defini comme un classifieur de severite a partir du dossier clinique complet.

## Experience 2 : RF sans diagnostics

Les colonnes suivantes sont exclues :

- `diagnosis`
- `diagnosis1`
- `diagnosis2`
- `diagnosis3`
- toutes les colonnes `diag_*`

Le modele utilise uniquement les variables cliniques structurees : age, douleur, signes cliniques, mouvements mandibulaires, antecedents, condyle, etc.

Resultats OOF CV=5 :

```text
Accuracy          : 0.9150
Balanced accuracy : 0.9011
Macro-F1          : 0.9081
Macro precision   : 0.9182
Macro recall      : 0.9011
```

Par classe :

```text
Mild    F1 = 0.92 | recall = 0.94 | n = 104
Normal  F1 = 0.99 | recall = 1.00 | n = 50
Severe  F1 = 0.81 | recall = 0.76 | n = 46
```

Matrice de confusion :

```text
             pred_Mild  pred_Normal  pred_Severe
true_Mild         98        1            5
true_Normal        0       50            0
true_Severe       11        0           35
```

Features les plus importantes :

```text
age
condyle_NA
condyle_B
condyle_L
age_visit
mmo_NA
click_B
click_NA
pain_scor
mio_NA
ttpp_TMJ_NA
teeth_grin_Y
```

Interpretation :

Le modele sans diagnostic reste tres performant. La perte de performance est faible :

```text
Avec diag_* : macro-F1 = 0.9179
Sans diag_* : macro-F1 = 0.9081
Difference : environ 0.0098 macro-F1
```

Cela montre que les variables cliniques structurees suffisent deja a separer correctement les trois classes. Les diagnostics ameliorent legerement la performance, mais ne sont pas indispensables.

## Experience 3 : blender CNN + RF sans nouvelle inference

Le blender reutilise :

- les probabilites CNN deja calculees dans `meta_oof_predictions.csv`
- les probabilites RF OOF dans `rf_cv5_oof_predictions.csv`

Aucune inference CNN n'est relancee.

Nombre de cas :

```text
Total lignes       : 200
Avec probas CNN    : 150
RF only            : 50
```

Les 50 cas `Normal` n'ont pas de probabilites CNN disponibles dans cette sortie CNN existante. Ils sont donc classes par la RF uniquement.

### Blender adaptatif calibre

Le script teste automatiquement les bornes de poids CNN. La meilleure calibration trouve :

```text
min_cnn_weight = 0.0
max_cnn_weight = 0.0
```

Donc, avec les probabilites CNN actuelles, le meilleur blender revient a utiliser uniquement la RF.

Resultats :

```text
Accuracy          : 0.9250
Balanced accuracy : 0.9075
Macro-F1          : 0.9179
```

### Blender force avec CNN = 0.6

Formule :

```text
prediction = 0.6 * CNN + 0.4 * RF
```

Resultats :

```text
Accuracy          : 0.8650
Balanced accuracy : 0.8448
Macro-F1          : 0.8474
Macro precision   : 0.8590
Macro recall      : 0.8448
```

Matrice de confusion :

```text
             pred_Mild  pred_Normal  pred_Severe
true_Mild         94        3            7
true_Normal        0       50            0
true_Severe       15        2           29
```

Interpretation :

Forcer le CNN a 0.6 degrade les performances, surtout pour la classe `Severe`. Les probabilites CNN disponibles semblent moins fiables que la RF clinique pour ce probleme 3 classes, ou bien elles ne sont pas parfaitement adaptees au contexte `Normal / Mild / Severe`.

## Conclusion

Le meilleur modele actuel est la Random Forest clinique avec features diagnostics normalisees :

```text
Macro-F1 = 0.9179
Accuracy = 0.9250
```

Cependant, la Random Forest sans diagnostics reste presque aussi bonne :

```text
Macro-F1 = 0.9081
Accuracy = 0.9150
```

Pour un rapport, il est donc recommande de presenter les deux resultats :

1. Modele complet post-evaluation clinique : avec `diag_*`.
2. Modele clinique plus generalisable : sans `diag_*`.

Le blender CNN/RF ne doit pas etre retenu avec un poids CNN fixe de 0.6, car il reduit les performances actuelles. Le code reste utile si de meilleures probabilites CNN 3 classes sont produites plus tard.

## Experience 4 : RF avec uniquement les donnees cliniques demandees

Une experience supplementaire a ete realisee avec uniquement les variables correspondant a la definition suivante :

- intensite de douleur
- bruits articulaires
- restrictions de mouvement
- mesures fonctionnelles : ouverture buccale, laterotrusion, protrusion
- deviation mandibulaire / dislocation
- informations demographiques : age, sexe, race
- antecedents medicaux et dentaires

Les variables exclues sont :

- diagnostics texte : `diagnosis`, `diagnosis1`, `diagnosis2`, `diagnosis3`
- features derivees du diagnostic : `diag_*`
- variables radiographiques non demandees, notamment `condyle`

Features utilisees :

```text
age, age_visit, age_range, gender, race,
pain_scor,
painarea_PA, painarea_P, painarea_T, painarea_L, painarea_R,
painarea_M, painarea_TMJ,
ttpp_TMJ, ttpp_M, ttpp_T, ttpp_P,
crep, click, dev, disl,
mmo, mio, lat_R, lat_L, maxprot,
teeth_wea, teeth_grin, hard_food,
MD_autoi, MD_CnR, MD_CP, MD_ment, MD_gastr,
MD_neuro, MD_infect, MD_derm
```

Protocole :

```text
Random Forest 3 classes
CV = 5
Seed = 42
Optuna = 5 trials par fold
Nombre de features brutes = 37
```

Resultats OOF CV=5 :

```text
Accuracy          : 0.7050
Balanced accuracy : 0.6775
Macro-F1          : 0.6691
Macro precision   : 0.6643
Macro recall      : 0.6775
```

Par classe :

```text
Mild    F1 = 0.73 | recall = 0.75 | n = 104
Normal  F1 = 0.96 | recall = 1.00 | n = 50
Severe  F1 = 0.31 | recall = 0.28 | n = 46
```

Matrice de confusion :

```text
             pred_Mild  pred_Normal  pred_Severe
true_Mild         78        2           24
true_Normal        0       50            0
true_Severe       31        2           13
```

Features les plus importantes :

```text
age
mmo_NA
click_NA
click_B
mio_NA
age_visit
pain_scor
ttpp_TMJ_NA
lat_R_NA
age_range_20-39
lat_L_NA
painarea_TMJ_NA
maxprot_WNL
teeth_grin_Y
```

Interpretation :

Ce modele est le plus strict cliniquement, car il exclut toute information issue du diagnostic et toute variable radiographique non incluse dans la definition demandee. Il separe tres bien les cas `Normal`, mais distingue beaucoup moins bien `Mild` et `Severe`. Cela suggere que, dans ce dataset, les informations les plus discriminantes pour separer `Mild` et `Severe` viennent soit des diagnostics normalises, soit des variables radiographiques/condylaires, plutot que des seuls symptomes et mesures cliniques disponibles.
