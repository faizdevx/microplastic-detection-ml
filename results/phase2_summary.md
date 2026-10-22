## Polymer identification: results (generated)

Selected on CV only: **hist_gb** `{'max_iter': 100, 'learning_rate': 0.1, 'max_leaf_nodes': 31}` on geometry `resample`.

### Cross-validation (5 group-aware folds, image level)

| Geometry | Model | Pooled accuracy | Fold mean ± std | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| resample | logreg | 0.972 | 0.972 ± 0.014 | 0.972 | 0.996 |
| resample | random_forest | 0.971 | 0.971 ± 0.021 | 0.971 | 0.995 |
| resample | hist_gb | 0.976 | 0.976 ± 0.013 | 0.976 | 0.997 |
| native | logreg | 0.940 | 0.940 ± 0.027 | 0.940 | 0.984 |
| native | random_forest | 0.940 | 0.940 ± 0.035 | 0.940 | 0.987 |
| native | hist_gb | 0.951 | 0.951 ± 0.022 | 0.951 | 0.991 |

### Controls (balanced accuracy unless noted)

| Geometry | Shortcut CV (size+intensity) | Res.-matched model / shortcut | 640→1280 model / shortcut | 1280→640 model / shortcut | Background-only | Permutation p | Passes |
|---|---|---|---|---|---|---|---|
| resample | 0.958 | 0.957 / 0.914 | 0.864 / 0.506 | 0.762 / 0.679 | 0.906 | 0.010 | no |
| native | 0.958 | 0.947 / 0.914 | 0.613 / 0.506 | 0.650 / 0.679 | 0.907 | 0.010 | no |

Feature-group ablation (HistGB, accuracy):

| Geometry | shape | ratio | intensity_norm | intensity_raw | all |
|---|---|---|---|---|---|
| resample | 0.868 | 0.862 | 0.953 | 0.983 | 0.978 |
| native | 0.732 | 0.871 | 0.931 | 0.977 | 0.953 |

### Hold-out (evaluated once)

| Metric | Value | 95 % CI (group bootstrap) |
|---|---|---|
| accuracy | 0.987 | 0.952 - 1.000 |
| balanced_accuracy | 0.988 | 0.958 - 1.000 |
| macro_f1 | 0.987 | 0.951 - 1.000 |
| roc_auc | 0.999 | 0.996 - 1.000 |

Confusion matrix (rows true PVC/NYLON): [[74, 0], [2, 79]]; ECE 0.125; 155 images.
