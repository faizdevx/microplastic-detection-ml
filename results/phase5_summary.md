## Phase 5: robustness of polymer identification (generated)

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

