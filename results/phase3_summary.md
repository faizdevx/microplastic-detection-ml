## Multi-angle scattering: model results (generated)

Config hash `23ce7b09ce75`; rig parameters are not measured, see `configs/scattering.yaml`.

### Concentration regression, fixed geometry (E1)

**by concentration level** (600 rows, ADC mcp3008):

| Model | R² (95 % CI) | RMSE mg/L | MAE mg/L | MAPE % | 90 % interval coverage |
|---|---|---|---|---|---|
| lasso | 0.807 (0.627 to 0.858) | 1.219 | 0.745 | 46.4 | 0.88 |
| ridge | 0.804 (0.629 to 0.850) | 1.228 | 0.746 | 46.3 | 0.88 |
| pls | 0.550 (0.174 to 0.787) | 1.862 | 0.936 | 52.2 | 0.91 |
| random_forest | 0.771 (0.512 to 0.837) | 1.329 | 0.834 | 47.8 | 0.93 |
| svr_rbf | 0.797 (0.578 to 0.862) | 1.249 | 0.787 | 43.5 | 0.93 |
| gaussian_process | 0.715 (0.576 to 0.836) | 1.481 | 0.842 | 46.1 | 0.90 |
| physics_inversion | 0.613 (0.466 to 0.728) | 1.727 | 0.896 | 45.5 | 0.92 |

**by replicate** (600 rows, ADC mcp3008):

| Model | R² (95 % CI) | RMSE mg/L | MAE mg/L | MAPE % | 90 % interval coverage |
|---|---|---|---|---|---|
| lasso | 0.847 (0.833 to 0.861) | 1.085 | 0.669 | 44.7 | 0.88 |
| ridge | 0.841 (0.828 to 0.850) | 1.107 | 0.681 | 44.7 | 0.89 |
| pls | 0.509 (0.348 to 0.691) | 1.945 | 0.974 | 50.8 | 0.87 |
| random_forest | 0.843 (0.832 to 0.852) | 1.098 | 0.652 | 42.7 | 0.88 |
| svr_rbf | 0.843 (0.833 to 0.853) | 1.101 | 0.671 | 39.4 | 0.88 |
| gaussian_process | 0.838 (0.826 to 0.853) | 1.116 | 0.679 | 43.0 | 0.89 |
| physics_inversion | 0.637 (0.539 to 0.704) | 1.672 | 0.873 | 45.1 | 0.90 |

![E1 regression](sensing/figures/regression_e1.png)

### Robustness: train on g0, test on perturbed geometries

R² of log10(concentration), re-baselined per geometry (blank level measured on each geometry); the linear-scale R² and CIs are in `results/phase3_regression.json`.

| Model | g1 | g2 | g3 | all_others |
|---|---|---|---|---|
| lasso | 0.83 | 0.63 | 0.81 | 0.76 |
| ridge | 0.83 | 0.45 | 0.80 | 0.69 |
| pls | 0.61 | 0.47 | 0.68 | 0.58 |
| random_forest | 0.83 | 0.76 | 0.82 | 0.80 |
| svr_rbf | 0.84 | 0.53 | 0.85 | 0.74 |
| gaussian_process | 0.71 | 0.58 | 0.80 | 0.70 |
| physics_inversion | 0.45 | 0.42 | 0.54 | 0.47 |

Same, but the blank level of the training geometry is applied to every geometry:

| Model | all_others |
|---|---|
| lasso | 0.20 |
| ridge | -0.45 |
| pls | -26.83 |
| random_forest | 0.74 |
| svr_rbf | -2.53 |
| gaussian_process | -0.00 |
| physics_inversion | -7.26 |

### PVC vs Nylon from scattering alone

d′ = |mean difference| / pooled noise, 400 readings per cell, 50 um, sigma_ln 0.3.

| Conc. mg/L | d′ S1 | d′ S2 | d′ S3 | d′ S4 | rel. diff S1 |
|---|---|---|---|---|---|
| 1 | 0.37 | 0.05 | 0.00 | 0.00 | -0.042 |
| 3 | 0.78 | 0.08 | 0.00 | 0.00 | -0.096 |
| 10 | 1.56 | 0.24 | 0.00 | 1.95 | -0.160 |

Differences at equal mass concentration come mostly from the density difference (different particle numbers per mg/L), not from the refractive index.

### Linearity and dynamic range

Optical depth reaches 0.1 at 279 mg/L. Saturation (10-bit ADC): {'S1': 44.56649301315341, 'S2': 885.1673390820464, 'S3': None, 'S4': None}.

### Detection limits (S1, S2)

| ADC | slope S1 counts per mg/L | LOD S1 mg/L | LOQ S1 mg/L | LOD S2 mg/L | LOQ S2 mg/L |
|---|---|---|---|---|---|
| mcp3008 | 23.45 | 0.432 | 1.309 | 0.146 | 0.441 |
| ads1115 | 604.63 | 0.429 | 1.300 | 0.447 | 1.355 |

LOD = 3.3 σ / slope and LOQ = 10 σ / slope, with σ the reading noise at 0.1 mg/L (particle-number noise included). Blank-only σ is below one count for the 10-bit ADC, which makes a blank-based LOD meaningless there (`analysis.json`).

### Geometry sensitivity

Span (max / min) of the S1 slope across the documented range of each parameter:

| Parameter | Span |
|---|---|
| tia_Mohm | 1000.0 |
| detector_distance_mm | 27.4 |
| laser_power_mW | 5.0 |
| path_length_mm | 4.0 |
| angle_shift_deg | 3.4 |

### Size dependence at fixed mass concentration

PVC at 1 mg/L, median size 20 µm to 80 µm: signal ratio per detector [0.251, 0.181, 0.145, 0.14].

