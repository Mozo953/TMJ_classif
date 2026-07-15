# ResNet14 + tabular RF blender

This folder recomputes a simple probability-level blender using only:

- ResNet14 OOF probabilities from `best_models/cnn_best_retrain_check/oof_resnet14.npy`
- Tabular Random Forest OOF probabilities from `best_models/rf_requested_clinical/rf_cv5_oof_predictions.csv`

Class order: `['Mild', 'Normal', 'Severe']`.

Best simple blend selected by accuracy, then balanced accuracy, then macro F1.

## Main results

| Model | ResNet14 weight | RF weight | Accuracy | Balanced accuracy | Macro F1 | Recall Mild | Recall Normal | Recall Severe | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet14 alone | 1.00 | 0.00 | 0.6867 | 0.6544 | 0.6255 | 0.8772 | 0.9000 | 0.1860 | 0.4701 |
| Tabular RF alone | 0.00 | 1.00 | 0.7000 | 0.6922 | 0.6884 | 0.6316 | 0.9800 | 0.4651 | 0.4034 |
| Blend 50/50 | 0.50 | 0.50 | 0.7333 | 0.7147 | 0.7067 | 0.7719 | 1.0000 | 0.3721 | 0.3938 |
| Best grid blend | 0.70 | 0.30 | 0.7533 | 0.7341 | 0.7245 | 0.8070 | 1.0000 | 0.3953 | 0.4140 |


## Meta-learner Logistic Regression OOF

Features: 3 ResNet14 probabilities + 3 tabular RF probabilities. Evaluation: 5-fold StratifiedKFold on the probability features.

| Model | Accuracy | Balanced accuracy | Macro F1 | Recall Mild | Recall Normal | Recall Severe | Brier | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Meta LR OOF | 0.6933 | 0.7016 | 0.6919 | 0.4737 | 0.9800 | 0.6512 | 0.3539 | 0.6977 |
