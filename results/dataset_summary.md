| Class | Files | Unique groups | Exact-duplicate files | Sessions | Resolutions |
|---|---|---|---|---|---|
| NYLON | 1084 | 315 | 542 | 8 | 742x556 (424), 1280x960 (200), 780x585 (128), 640x480 (96) |
| PVC | 956 | 285 | 478 | 2 | 640x480 (764), 1280x960 (192) |

Near-duplicate threshold: pHash Hamming distance <= 6.

| Shortcut baseline (group-aware 5-fold CV) | Logistic regression | Random forest |
|---|---|---|
| size_only | 0.753 ± 0.040 | 0.883 ± 0.019 |
| size_and_intensity | 0.895 ± 0.035 | 0.971 ± 0.012 |

Metadata/intensity-only classifiers reach 0.971 group-aware CV accuracy (threshold 0.7): class is predictable without looking at particles.
