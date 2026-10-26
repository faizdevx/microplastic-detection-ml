"""Concentration to detector power: size distributions, attenuation, multiple scattering."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from mp.scatter import mie

TH_GRID = np.concatenate([np.linspace(0.0, 4.0, 801), np.linspace(4.0, 180.0, 1761)[1:]])


@dataclass
class Rig:
    """Everything about the optical train that the model needs."""

    theta_deg: tuple = (20.0, 45.0, 90.0, 135.0)
    distance_m: float = 0.05
    area_m2: float = 7.5e-6
    path_m: float = 0.01
    beam_diameter_m: float = 1e-3
    independent_volumes: float = 100.0
    laser_W: float = 5e-3
    wavelength_nm: float = 650.0
    n_medium: float = 1.331
    responsivity: float = 0.4
    dark_current_A: float = 2e-9
    tia_ohm: float = 1e8
    bandwidth_Hz: float = 10.0
    stray_fraction: float = 1e-8
    ambient_offset_V: float = 2e-3
    gain_scale: float = 1.0

    @property
    def solid_angle(self) -> float:
        return self.area_m2 / self.distance_m**2

    @property
    def half_angle_deg(self) -> float:
        return float(np.degrees(np.sqrt(self.solid_angle / np.pi)))

    @property
    def illuminated_volume_m3(self) -> float:
        return np.pi * (self.beam_diameter_m / 2) ** 2 * self.path_m


def rig_from_config(cfg: dict) -> Rig:
    g, las, d, med = cfg["geometry"], cfg["laser"], cfg["detector"], cfg["medium"]
    return Rig(
        theta_deg=tuple(float(t) for t in g["theta_deg"]),
        distance_m=g["distance_mm"] * 1e-3,
        area_m2=d["area_mm2"] * 1e-6,
        path_m=g["path_length_mm"] * 1e-3,
        beam_diameter_m=g["beam_diameter_mm"] * 1e-3,
        independent_volumes=float(g["independent_volumes"]),
        laser_W=las["power_mW"] * 1e-3,
        wavelength_nm=float(las["wavelength_nm"]),
        n_medium=med["n"],
        responsivity=d["responsivity_A_W"],
        dark_current_A=d["dark_current_nA"] * 1e-9,
        tia_ohm=d["tia_Mohm"] * 1e6,
        bandwidth_Hz=d["bandwidth_Hz"],
        stray_fraction=d["stray_light_fraction"],
        ambient_offset_V=d["ambient_offset_mV"] * 1e-3,
    )


def lognormal_nodes(d_median_um: float, sigma_ln: float, n: int = 40) -> np.ndarray:
    """Equal-probability diameters (um) of a number-weighted log-normal distribution."""
    z = norm.ppf((np.arange(n) + 0.5) / n)
    return d_median_um * np.exp(sigma_ln * z)


def number_density(conc_mg_L: float, density_g_cm3: float, d_um: np.ndarray) -> float:
    """Particles per m^3 for a mass concentration. 1 mg/L = 1 g/m^3.

    N = C / (rho * pi/6 * <d^3>), <d^3> averaged over the size nodes.
    """
    d_m = np.asarray(d_um) * 1e-6
    mean_vol = np.pi / 6 * np.mean(d_m**3)
    return conc_mg_L / (density_g_cm3 * 1e6 * mean_vol)


@dataclass
class ParticleOptics:
    """Size-averaged optical properties of one polymer + size distribution."""

    sigma_ext: float                      # m^2, mean over particles
    sigma_sca: float
    g: float                              # sigma_sca-weighted asymmetry
    dsigma_dOmega: np.ndarray             # m^2/sr at each detector angle (acceptance-averaged)
    theta_grid_deg: np.ndarray = field(repr=False, default=None)
    phase: np.ndarray = field(repr=False, default=None)   # sigma_sca-weighted p(theta), 1/sr


def particle_optics(
    n_p: float, k_p: float, d_nodes_um: np.ndarray, rig: Rig, with_phase: bool = False
) -> ParticleOptics:
    m = mie.relative_index(n_p, k_p, rig.n_medium)
    k_med = 2 * np.pi * rig.n_medium / (rig.wavelength_nm * 1e-9)
    alpha = rig.half_angle_deg
    offsets = np.linspace(-alpha, alpha, 5)
    th = np.asarray(rig.theta_deg, float)
    th_all = np.clip(th[:, None] + offsets[None, :], 0.05, 179.95)
    sig_ext = sig_sca = g_num = 0.0
    dsig = np.zeros(len(th))
    phase = np.zeros_like(TH_GRID) if with_phase else None
    for d in d_nodes_um:
        x = float(mie.size_parameter(d, rig.wavelength_nm, rig.n_medium))
        cs = mie.cross_sections(m, x)
        area = np.pi * (d * 1e-6 / 2) ** 2
        s_ext, s_sca = cs["q_ext"] * area, cs["q_sca"] * area
        sig_ext += s_ext
        sig_sca += s_sca
        g_num += cs["g"] * s_sca
        inten = mie.angular_intensity(m, x, th_all.ravel())["unpolarised"].reshape(th_all.shape)
        dsig += inten.mean(axis=1) / k_med**2
        if with_phase:
            phase += s_sca * mie.phase_function(m, x, TH_GRID)
    n = len(d_nodes_um)
    return ParticleOptics(
        sigma_ext=sig_ext / n, sigma_sca=sig_sca / n, g=g_num / sig_sca,
        dsigma_dOmega=dsig / n, theta_grid_deg=TH_GRID if with_phase else None,
        phase=(phase / sig_sca) if with_phase else None,
    )


def detector_power(
    conc_mg_L: float, popt: ParticleOptics, rig: Rig, density_g_cm3: float,
    d_nodes_um: np.ndarray, mode: str = "first_order", mc_phase=None,
    mc_cfg: dict | None = None, seed: int = 0,
) -> dict:
    """Mean optical power (W) on each detector plus the optical depth.

    none         single scattering, Beer-Lambert attenuation with sigma_ext
    first_order  single scattering, attenuation with the transport cross section
                 sigma_ext - g sigma_sca (forward-scattered light is not lost)
    monte_carlo  photon transport in a square cuvette with next-event detector estimates
    """
    n_dens = number_density(conc_mg_L, density_g_cm3, d_nodes_um)
    mu_ext = n_dens * popt.sigma_ext
    tau = mu_ext * rig.path_m
    if mode in ("none", "first_order"):
        mu_att = mu_ext if mode == "none" else n_dens * (popt.sigma_ext - popt.g * popt.sigma_sca)
        p = (rig.laser_W * n_dens * rig.path_m * popt.dsigma_dOmega * rig.solid_angle
             * np.exp(-mu_att * rig.path_m))
    elif mode == "monte_carlo":
        from mp.scatter import mc

        p = mc.detector_power(n_dens, popt, rig, mc_phase if mc_phase is not None
                              else (popt.theta_grid_deg, popt.phase), mc_cfg or {}, seed)
    else:
        raise ValueError(f"unknown multiple_scattering mode {mode!r}")
    return {"P_W": np.asarray(p), "tau": float(tau), "n_per_m3": float(n_dens)}
