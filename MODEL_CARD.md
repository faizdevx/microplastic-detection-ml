# Model Card — polymer identification (PVC vs Nylon), baseline A

All numbers are generated (`results/phase2_*.json`). The hold-out was evaluated once, at git
commit `00cd56b2` (the parent of the commit that added the Phase 2 code).

## Intended use
Research prototype that labels Nile Red fluorescence microscope images of **pristine,
laboratory-prepared PVC or Nylon powder** as one of the two polymers. It is a study of what
such a classifier can learn from this dataset, not a field instrument.

## Not for
- Environmental or weathered samples, biofouled or UV-aged particles.
- Any polymer other than PVC and Nylon. There is no "other" class, so an unknown particle is
  forced into one of the two.
- Any claim about chemical identity. Fluorescence colour and brightness are indirect.

## Training data
- `RAW_DATASET.zip` (Zenodo 10.5281/zenodo.21073653): 2040 files, 600 near-duplicate groups,
  1020 unique files after dropping byte-identical copies. PVC and Nylon only. See `DATASET_CARD.md`.
- Splits: group-level hold-out (155 unique images) and 5 group-aware CV folds
  from `data/processed/splits.json`.
- Labels: image-level (class of the folder). Per-particle labels are inherited.

## Model
Per-particle features from a background-corrected red channel (Otsu threshold, watershed),
mean probability over particles per image. Selected on CV only: `hist_gb`,
`{'max_iter': 100, 'learning_rate': 0.1, 'max_leaf_nodes': 31}`, geometry `resample` (every image letterboxed to 640×480).
Pixel size is uncalibrated (`um_per_px: null`), so all sizes are in pixels.

## Metrics
Cross-validation: pooled image-level accuracy 0.976
(fold mean 0.976 ± 0.013), ROC-AUC 0.997.

Hold-out (155 images, evaluated once): accuracy 0.987
(95 % CI 0.952 to 1.000, group bootstrap), macro-F1 0.987,
ROC-AUC 0.999, confusion matrix (rows true PVC, Nylon) [[74, 0], [2, 79]], ECE 0.125.
Reliability plot: `results/figures/phase2_reliability.png`. The failure gallery in
`results/figures/phase2_failures/` shows all 2 hold-out errors (fewer than 12 exist).

## The confound and what the controls say
Phase 1 found that image size and global colour statistics alone predict the class
(0.971 accuracy, random forest, group-aware CV). The controls test whether the particle
classifier is more than that.

| Control (resample geometry) | Model | Shortcut |
|---|---|---|
| Resolution-matched subset, balanced accuracy | 0.957 | 0.914 |
| Train 640×480, test 1280×960 | 0.864 | 0.506 |
| Train 1280×960, test 640×480 (320 PVC / 36 Nylon) | 0.762 | 0.679 |
| Background only, particles masked (chance 0.5) | 0.906 | |
| Label permutation, 100 permutations | p = 0.010 (null mean 0.499) | |

Feature groups alone (accuracy): shape 0.868, colour ratios
0.862, normalised intensity 0.953,
raw intensity 0.983. With `native` geometry (no resizing) the
cross-resolution balanced accuracies fall to 0.613
and 0.650.

**Verdict.** The model beats the metadata shortcut under `resample` geometry, so some of the
signal is in the particles. It **fails the background-only control** (0.906 against a chance
level of 0.5), so the pass rule is not met. The permutation test (p = 0.010) shows the
signal is not noise; it does not show the signal is chemical. The two classes differ in
illumination, exposure and noise outside the particles, and raw intensity dominates the
feature importance (`results/phase2_explain.json`). Treat the accuracy as an upper bound for
this dataset and acquisition setup. Imaging both polymers under identical settings is the
clean fix.

## Failure modes
- Touching or overlapping particles are split or merged by the watershed.
- Faint particles below the signal floor are missed; border-touching particles are dropped.
- Sensitive to exposure and illumination (raw intensity features).
- Probabilities are not well calibrated (ECE 0.125).

## Not evaluated
- **YOLOv11-seg:** only a CPU smoke test was run (2 epochs, 10 images,
  mask mAP50 0.00, which is not a result). No per-fold training, no mAP, no
  McNemar comparison, no counting error. This environment has CPU-only PyTorch.
- **Human-corrected masks:** none imported; masks are auto-generated.
- **Pixel calibration:** none.
