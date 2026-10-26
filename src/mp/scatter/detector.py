"""Photodiode, transimpedance amplifier and ADC."""
from __future__ import annotations

import numpy as np

Q_E = 1.602176634e-19
K_B = 1.380649e-23


def adc_counts(volts: np.ndarray, bits: int, full_scale_V: float) -> tuple[np.ndarray, np.ndarray]:
    """Quantise to [0, 2^bits - 1]. Returns (counts, saturated_mask)."""
    top = 2**bits - 1
    raw = np.floor(np.asarray(volts) / full_scale_V * 2**bits)
    return np.clip(raw, 0, top).astype(int), raw >= top


def readings(
    power_W: np.ndarray, rig, n_per_m3: float, adc: dict, noise: dict,
    drift_pct: float, temperature_K: float, rng: np.random.Generator,
) -> dict:
    """One reading of all detectors from the mean optical power on each of them."""
    power_W = np.asarray(power_W, float)
    scale = 1.0
    if noise.get("particle_number", True) and n_per_m3 > 0:
        mean_count = n_per_m3 * rig.illuminated_volume_m3 * rig.independent_volumes
        scale *= rng.poisson(mean_count) / mean_count
    drift = 1.0 + (rng.normal(0.0, drift_pct / 100.0) if noise.get("laser_drift", True) else 0.0)
    p = power_W * scale * drift + rig.stray_fraction * rig.laser_W * drift

    i_dark = rig.dark_current_A if noise.get("dark", True) else 0.0
    i_mean = rig.responsivity * p + i_dark
    sigma2 = np.zeros_like(i_mean)
    bw = rig.bandwidth_Hz
    if noise.get("shot", True):
        sigma2 += 2 * Q_E * i_mean * bw
    if noise.get("johnson", True):
        sigma2 += 4 * K_B * temperature_K * bw / rig.tia_ohm
    current = i_mean + rng.normal(0.0, np.sqrt(sigma2))
    volts = current * rig.tia_ohm * rig.gain_scale
    if noise.get("ambient", True):
        volts = volts + rig.ambient_offset_V * (1.0 + 0.05 * rng.normal(size=volts.shape))
    counts, sat = adc_counts(volts, adc["bits"], adc["full_scale_V"])
    return {"P_W": p, "V": volts, "S": counts, "saturated": sat}
