# Rig geometry used by the scattering model

Parameters in `configs/scattering.yaml`. **None of these values has been measured** on a
built rig; they are placeholders with the ranges below, and the analyses sweep them.

| Parameter | Default | Range swept | Status |
|---|---|---|---|
| Detector angles θ1..θ4 | 20°, 45°, 90°, 135° | ±5° shift | placeholder |
| Detector distance | 50 mm | 20-100 mm | placeholder |
| Detector area | 7.5 mm² (BPW34) | fixed | datasheet value, to verify |
| Cuvette path length | 10 mm | 5-20 mm | placeholder |
| Beam diameter | 1 mm | 0.5-3 mm | placeholder |
| Independent volumes per reading | 100 | 10-1000 | placeholder |
| Laser power | 5 mW | 1-5 mW | placeholder |
| Transimpedance R_f | 100 Mohm | 1 Mohm-1 Gohm | placeholder |
| ADC | MCP3008, 10-bit, 3.3 V | MCP3008, ADS1115 | placeholder |
| Stray light fraction | 1e-8 of laser power | 1e-9 to 1e-6 | placeholder |

## Which values matter most

Span (maximum / minimum) of the S1 slope, in counts per mg/L, across the swept range of one
parameter at a time (`results/sensing/analysis.json`):

| Parameter | Span |
|---|---|
| tia_Mohm | 1000.0 |
| detector_distance_mm | 27.4 |
| laser_power_mW | 5.0 |
| path_length_mm | 4.0 |
| angle_shift_deg | 3.4 |

At the default values the slope of S1 is 23.5 counts per mg/L; S3 and S4 (90° and
135°) carry 0.03 and 0.07 counts per mg/L, which is below the noise.
