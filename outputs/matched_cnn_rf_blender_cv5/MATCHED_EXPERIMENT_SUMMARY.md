# Image-matched clinical RF + CNN/RF blender CV=5

## Filtrage strict

Seuls les cas ayant a la fois :

- une ligne clinique dans `clinical_merged_with_diag_features.csv`
- une prediction/probabilite CNN deja calculee dans `meta_oof_predictions.csv`

ont ete gardes.

Les cas cliniques sans image/proba CNN ont ete exclus. Les images sans cas clinique auraient aussi ete exclues.

Resultat :

```text
Clinical cases initiaux : 200
Cas avec clinical + CNN : 150
Clinical sans CNN       : 50
CNN sans clinical       : 0
```

Repartition finale :

```text
Mild   : 56
Normal : 50
Severe : 44
```

Un seul desaccord de label a ete trouve entre le label CNN et le label clinique :

```text
Case ID 35 : clinical = Severe, CNN label = Mild
```

Le label clinique `severity` a ete utilise comme verite terrain finale.

## RF clinique reentrainee sur les cas matches

Protocole :

```text
Random Forest 3 classes
CV = 5
Seed = 42
Optuna = 5 trials par fold
```

Resultats OOF :

```text
Accuracy          : 0.9333
Balanced accuracy : 0.9242
Macro-F1          : 0.9299
Macro precision   : 0.9495
Macro recall      : 0.9242
```

Matrice de confusion :

```text
             pred_Mild  pred_Normal  pred_Severe
true_Mild         56        0            0
true_Normal        0       50            0
true_Severe       10        0           34
```

## CNN seul sur les cas matches

Les probabilites CNN sont reutilisees directement, sans inference.

```text
Accuracy          : 0.6733
Balanced accuracy : 0.6607
Macro-F1          : 0.6527
Macro precision   : 0.6547
Macro recall      : 0.6607
```

## Meta-blender CNN + RF reentraine en CV=5

Le meta-blender utilise comme entrees :

```text
cnn_prob_Mild, cnn_prob_Normal, cnn_prob_Severe
rf_prob_Mild,  rf_prob_Normal,  rf_prob_Severe
```

Modele :

```text
StandardScaler + LogisticRegression(class_weight='balanced')
CV = 5
Seed = 42
```

Resultats OOF :

```text
Accuracy          : 0.8667
Balanced accuracy : 0.8640
Macro-F1          : 0.8651
Macro precision   : 0.8666
Macro recall      : 0.8640
```

Matrice de confusion :

```text
             pred_Mild  pred_Normal  pred_Severe
true_Mild         47        1            8
true_Normal        1       49            0
true_Severe       10        0           34
```

## Simple weighted blender

Un test de poids fixe a aussi ete fait :

```text
blend = w * CNN + (1 - w) * RF
```

Le meilleur poids trouve est :

```text
w = 0.0
```

Ce qui signifie que le meilleur weighted blender revient a utiliser la RF seule.

## Conclusion

Sur les 150 cas strictement associes a une image et a une ligne clinique, la RF clinique est le meilleur modele :

```text
RF seule        : macro-F1 = 0.9299
CNN seul        : macro-F1 = 0.6527
Meta CNN + RF   : macro-F1 = 0.8651
Best weighted   : macro-F1 = 0.9299 avec poids CNN = 0
```

Le CNN degrade le score lorsqu'il est force dans le blender actuel. La meilleure strategie avec les probabilites CNN disponibles est donc de conserver la RF clinique seule pour cette evaluation matchee.
