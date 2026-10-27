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
- **YOLOv11-seg:** only a CPU smoke test was run (2 epochs, 10 images; the metrics of that run are not a result). No per-fold training, no mAP, no
  McNemar comparison, no counting error. This environment has CPU-only PyTorch.
- **Human-corrected masks:** none imported; masks are auto-generated.
- **Pixel calibration:** none.

## Phase 5: robustness analysis

Cross-validation pool only; the hold-out was not read. `resample` geometry, 865 images.

| Variant | CV accuracy | 640→1280 bal. acc. | 1280→640 bal. acc. | Res.-matched bal. acc. | Background discs bal. acc. | Permutation p |
|---|---|---|---|---|---|---|
| baseline | 0.979 | 0.864 | 0.762 | 0.966 | 0.811 | 0.032 |
| V1_local_normalisation | 0.994 | 0.941 | 0.941 | 0.988 | 0.940 | 0.032 |
| V2_exposure_randomisation | 0.971 | 0.810 | 0.824 | 0.951 | 0.811 | 0.032 |
| V3_patch_cnn | 0.747 | 0.500 | 0.830 | 0.834 | 0.659 | 0.091 |

Background discs: discs of class-independent size placed on pixels that touch no particle; the same pipeline is trained and tested on them. Chance is 0.5. The Phase 2 image-level background control gave 0.906.

### Acquisition probes (higher than chance = the representation carries the cue)

| Variant | Exposure, overall | Exposure, within class | Resolution | Mean abs. Spearman ρ, P(NYLON) vs brightness |
|---|---|---|---|---|
| baseline | 0.850 (chance 0.334) | 0.754 (chance 0.335) | 0.854 (chance 0.425) | 0.397 |
| V1_local_normalisation | 0.731 (chance 0.334) | 0.663 (chance 0.335) | 0.903 (chance 0.425) | 0.302 |
| V2_exposure_randomisation | 0.850 (chance 0.334) | 0.754 (chance 0.335) | 0.854 (chance 0.425) | 0.489 |
| V3_patch_cnn | 0.601 (chance 0.334) | 0.490 (chance 0.335) | 0.685 (chance 0.425) | 0.305 |

### Verdict against the baseline

Thresholds: cross-resolution balanced accuracy must rise by at least 0.02, background-disc balanced accuracy must fall by at least 0.05.

| Variant | Cross-resolution | Background discs | Overall |
|---|---|---|---|
| V1_local_normalisation | improves (+0.128) | worse (+0.130) | worse |
| V2_exposure_randomisation | no change (+0.004) | no change (+0.000) | no clear improvement |
| V3_patch_cnn | worse (-0.148) | improves (-0.152) | worse |

### Which features carry the signal (same model, one feature set at a time)

| Feature set | CV accuracy | 640→1280 bal. acc. | 1280→640 bal. acc. | Background discs bal. acc. |
|---|---|---|---|---|
| shape | 0.864 | 0.582 | 0.781 | 0.516 |
| colour_ratios | 0.867 | 0.655 | 0.719 | 0.714 |
| shape+colour_ratios | 0.905 | 0.763 | 0.758 | 0.709 |
| normalised_intensity | 0.954 | 0.804 | 0.775 | 0.808 |
| raw_intensity | 0.986 | 0.679 | 0.763 | 0.890 |
| v1_local | 0.994 | 0.947 | 0.941 | 0.938 |

Shape features alone give CV accuracy 0.864 and cross-resolution balanced accuracy 0.582 / 0.781; on background discs they sit at 0.516, as they should because the discs have no class-dependent shape. Every feature set built from pixel values separates the classes on background discs too (0.709 to 0.938), so their accuracy mixes particle and background information.

No variant improves on both controls; the Phase 2 model stays the default in `mp analyse`.


Analyses that used the cross-validation pool only: all Phase 5 variants, controls, probes and the verdict. The hold-out (`results/phase2_holdout.json`) was evaluated once in Phase 2 and not touched again.
