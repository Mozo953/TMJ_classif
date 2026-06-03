# Justification accuracy - blender CNN + RF requested clinical

Critere principal: accuracy OOF CV=5.

Meilleure combinaison par accuracy:

- CNN weight: 0.50
- RF requested clinical weight: 0.50
- Accuracy: 0.7200
- Balanced accuracy: 0.7087
- Macro-F1: 0.7039

Interpretation: la combinaison gagnante selon l'accuracy est un blend equilibre CNN/RF. Elle depasse le RF requested clinical seul et le CNN seul dans cette grille de ponderation.
