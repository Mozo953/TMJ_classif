# ResNet14 embedding heads and PCA

This folder contains analyses where ResNet14 is used as a frozen feature extractor and small classifiers are trained on its embeddings.

## Embedding definitions

| Embedding | Definition | Dimension |
|---|---|---:|
| `penultimate_layer2_gap` | output of ResNet14 `layer2` after global average pooling | 32 |
| `last_layer3_gap` | output of ResNet14 `layer3` after global average pooling, before original FC | 64 |

## Heads tested

| Head | Description |
|---|---|
| `logistic_regression` | StandardScaler + class-balanced Logistic Regression |
| `linear_layer` | StandardScaler + one PyTorch linear layer |
| `light_mlp` | StandardScaler + MLP with one hidden layer |

## Results

| Embedding | Head | Accuracy | Balanced accuracy | Macro F1 | Recall Mild | Recall Normal | Recall Severe |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline ResNet14 | original FC | 0.6867 | 0.6544 | 0.6255 | 0.8772 | 0.9000 | 0.1860 |
| layer3 GAP | Logistic Regression | 0.6467 | 0.6528 | 0.6466 | 0.4737 | 0.8800 | 0.6047 |
| layer3 GAP | Linear layer | 0.6067 | 0.6082 | 0.6032 | 0.4561 | 0.8800 | 0.4884 |
| layer3 GAP | MLP | 0.5800 | 0.5695 | 0.5593 | 0.5263 | 0.8800 | 0.3023 |
| layer2 GAP | Logistic Regression | 0.5800 | 0.5829 | 0.5782 | 0.4035 | 0.8800 | 0.4651 |
| layer2 GAP | Linear layer | 0.6000 | 0.5947 | 0.5962 | 0.5088 | 0.8800 | 0.3953 |
| layer2 GAP | MLP | 0.6000 | 0.5863 | 0.5770 | 0.5965 | 0.8600 | 0.3023 |

## PCA

PCA outputs are in:

```text
pca_layers2_3/
```

| Embedding | PC1 | PC2 | PC1 + PC2 |
|---|---:|---:|---:|
| layer2 penultimate GAP | 29.1% | 26.8% | 55.9% |
| layer3 last GAP | 27.9% | 18.3% | 46.3% |

## Reproduce

```powershell
python .\code_classification\resnet14_embedding_heads_cv.py
python .\code_classification\make_resnet14_embedding_pca.py
```
