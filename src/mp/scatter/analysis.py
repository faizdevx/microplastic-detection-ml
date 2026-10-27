"""Analyses of the rig model: angular curves, polymer contrast, linearity, geometry
sensitivity, size dependence and detection limits."""
from __future__ import annotations

import json
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mp.config import resolve  # noqa: E402
from mp.eval.metrics import run_meta  # noqa: E402
from mp.scatter import detector as det  # noqa: E402
from mp.scatter import generate as G  # noqa: E402
from mp.scatter import mie  # noqa: E402
from mp.scatter import optics as O  # noqa: E402

OUT = "results/sensing"
D_REF = {"d_median_um": 50.0, "sigma_ln": 0.3}


def out_dir():
    d = resolve(OUT)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def mean_counts(P_W, rig, adc) -> np.ndarray:
    """Expected ADC counts (unquantised, unclipped) for mean optical power on each detector."""
    p = np.asarray(P_W) + rig.stray_fraction * rig.laser_W
    v = (rig.responsivity * p + rig.dark_current_A) * rig.tia_ohm * rig.gain_scale
    v = v + rig.ambient_offset_V
    return v / adc["full_scale_V"] * 2 ** adc["bits"]


_POPT: dict = {}


def _popt(cfg, rig, polymer, d_med, sigma):
    """Particle optics depend on angles, distance and wavelength only; cache them."""
    key = (polymer, d_med, sigma, rig.theta_deg, round(rig.distance_m, 9), rig.area_m2,
           rig.wavelength_nm, rig.n_medium)
    if key not in _POPT:
        nodes = O.lognormal_nodes(d_med, sigma, cfg["size_distribution"]["nodes"])
        p = cfg["polymers"][polymer]
        _POPT[key] = (nodes, O.particle_optics(p["n"], p["k"], nodes, rig))
    return _POPT[key]


def _power(cfg, rig, polymer, d_med, sigma, conc, mode="first_order"):
    nodes, popt = _popt(cfg, rig, polymer, d_med, sigma)
    return O.detector_power(conc, popt, rig, cfg["polymers"][polymer]["density_g_cm3"],
                            nodes, mode)


def _blank_counts(rig, adc):
    return mean_counts(np.zeros(4), rig, adc)


def signal_counts(cfg, rig, adc, polymer, d_med, sigma, conc, mode="first_order") -> np.ndarray:
    """Counts above the blank level."""
    r = _power(cfg, rig, polymer, d_med, sigma, conc, mode)
    return mean_counts(r["P_W"], rig, adc) - _blank_counts(rig, adc)


# --------------------------------------------------------------------------------------
def angular_curves(cfg, rig) -> dict:
    theta = np.concatenate([np.linspace(0.1, 6, 300), np.linspace(6, 180, 600)[1:]])
    k = 2 * np.pi * rig.n_medium / (rig.wavelength_nm * 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    at_det = {}
    for ax, d in zip(axes, (20.0, 50.0, 90.0)):
        x = float(mie.size_parameter(d, rig.wavelength_nm, rig.n_medium))
        for pol, style in (("PVC", "-"), ("NYLON", "--")):
            p = cfg["polymers"][pol]
            m = mie.relative_index(p["n"], p["k"], rig.n_medium)
            dsig = mie.angular_intensity(m, x, theta)["unpolarised"] / k**2
            ax.semilogy(theta, dsig, style, label=pol, lw=1)
            at_det.setdefault(f"d{int(d)}um", {})[pol] = [
                float(v) for v in mie.angular_intensity(m, x, rig.theta_deg)["unpolarised"] / k**2]
        for t in rig.theta_deg:
            ax.axvline(t, color="gray", lw=0.6, ls=":")
        ax.set_title(f"d = {d:.0f} um (x = {x:.0f})")
        ax.set_xlabel("scattering angle (deg)")
    axes[0].set_ylabel("dσ/dΩ (m²/sr)")
    axes[0].legend()
    fig.suptitle("Angular scattering of PVC and Nylon spheres in water at 650 nm; dotted = detector angles")
    fig.tight_layout()
    fig.savefig(out_dir() / "figures" / "angular_curves.png", dpi=120)
    plt.close(fig)
    return {"dsigma_dOmega_at_detector_angles_m2_per_sr": at_det, "theta_deg": list(rig.theta_deg)}


def polymer_contrast(cfg, rig, n_read=400, seed=11) -> dict:
    adc = G.adc_settings(cfg)
    rng = np.random.default_rng(seed)
    rows = {}
    for conc in (1.0, 3.0, 10.0):
        pw = {}
        for pol in ("PVC", "NYLON"):
            r = _power(cfg, rig, pol, **{"d_med": 50.0, "sigma": 0.3}, conc=conc)
            reads = np.array([
                det.readings(r["P_W"], rig, r["n_per_m3"], adc, cfg["noise"],
                             cfg["laser"]["drift_pct"], cfg["medium"]["temperature_K"], rng)["S"]
                for _ in range(n_read)], float)
            pw[pol] = reads
        mean_d = pw["PVC"].mean(0) - pw["NYLON"].mean(0)
        pooled = np.sqrt(0.5 * (pw["PVC"].var(0, ddof=1) + pw["NYLON"].var(0, ddof=1)))
        rel = mean_d / (0.5 * (pw["PVC"].mean(0) + pw["NYLON"].mean(0)))
        rows[f"{conc:g}"] = {
            "mean_diff_counts": mean_d.round(3).tolist(),
            "noise_std_counts": pooled.round(3).tolist(),
            "snr_d_prime": (np.abs(mean_d) / np.maximum(pooled, 1e-9)).round(4).tolist(),
            "relative_difference": rel.round(4).tolist(),
        }
    # polymer contrast of the noise-free forward-detector ratio S1/S2 (size fixed)
    return {"n_readings": n_read, "size": "50 um, sigma_ln 0.3", "by_concentration_mg_L": rows,
            "max_snr_d_prime": float(max(max(v["snr_d_prime"]) for v in rows.values()))}


def linearity(cfg, rig) -> dict:
    adc = G.adc_settings(cfg)
    conc = np.geomspace(0.05, 2000, 40)
    nodes = O.lognormal_nodes(50.0, 0.3, cfg["size_distribution"]["nodes"])
    p = cfg["polymers"]["PVC"]
    popt = O.particle_optics(p["n"], p["k"], nodes, rig)
    sig = []
    taus = []
    for c in conc:
        r = O.detector_power(c, popt, rig, p["density_g_cm3"], nodes, "first_order")
        sig.append(mean_counts(r["P_W"], rig, adc) - _blank_counts(rig, adc))
        taus.append(r["tau"])
    sig, taus = np.array(sig), np.array(taus)
    c_tau = float(np.interp(0.1, taus, conc))
    top = 2**adc["bits"] - 1
    base = _blank_counts(rig, adc)
    sat = {}
    for i in range(4):
        above = np.where(sig[:, i] + base[i] >= top)[0]
        sat[f"S{i + 1}"] = float(conc[above[0]]) if len(above) else None
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, t in enumerate(rig.theta_deg):
        ax.loglog(conc, np.maximum(sig[:, i], 1e-3), label=f"S{i + 1} ({t:g}°)")
    ax.axvline(c_tau, color="k", ls="--", lw=0.8, label="τ = 0.1")
    ax.axhline(top, color="r", ls=":", lw=0.8, label="ADC full scale")
    ax.set_xlabel("concentration (mg/L)")
    ax.set_ylabel("counts above blank (noise-free)")
    ax.set_title("Detector signal vs concentration (PVC, 50 µm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir() / "figures" / "linearity.png", dpi=120)
    plt.close(fig)
    low = (conc >= 0.1) & (conc <= 10)
    slope = [float(np.polyfit(conc[low], sig[low, i], 1)[0]) for i in range(4)]
    r2 = [float(np.corrcoef(conc[low], sig[low, i])[0, 1] ** 2) for i in range(4)]
    return {"conc_at_tau_0.1_mg_L": c_tau, "saturation_conc_mg_L": sat,
            "slope_counts_per_mg_L_0.1_to_10": slope, "r2_linear_0.1_to_10": r2,
            "blank_counts": base.round(2).tolist()}


def _slope(cfg, rig, adc, polymer="PVC", d_med=50.0, c=1.0) -> np.ndarray:
    """Counts per mg/L at low concentration for each detector."""
    return signal_counts(cfg, rig, adc, polymer, d_med, 0.3, c) / c


def geometry_sensitivity(cfg, rig) -> dict:
    adc = G.adc_settings(cfg)
    base_slope = _slope(cfg, rig, adc)
    sweeps = {
        "tia_Mohm": [(1.0, 1000.0), lambda r, v: replace(r, tia_ohm=v * 1e6)],
        "detector_distance_mm": [(20.0, 100.0), lambda r, v: replace(r, distance_m=v * 1e-3)],
        "path_length_mm": [(5.0, 20.0), lambda r, v: replace(r, path_m=v * 1e-3)],
        "laser_power_mW": [(1.0, 5.0), lambda r, v: replace(r, laser_W=v * 1e-3)],
        "angle_shift_deg": [(-5.0, 5.0), lambda r, v: replace(
            r, theta_deg=tuple(float(np.clip(t + v, 5, 175)) for t in r.theta_deg))],
    }
    out, rank = {}, {}
    fig, axes = plt.subplots(1, len(sweeps), figsize=(3.2 * len(sweeps), 3.4), sharey=True)
    for ax, (name, (rng_, fn)) in zip(axes, sweeps.items()):
        vals = np.geomspace(*rng_, 9) if rng_[0] > 0 else np.linspace(*rng_, 9)
        sl = np.array([_slope(cfg, fn(rig, v), adc) for v in vals])
        top = 2**adc["bits"] - 1
        sat_c = []
        for v in vals:
            r2 = fn(rig, v)
            conc = np.geomspace(0.05, 2000, 60)
            sg = np.array([signal_counts(cfg, r2, adc, "PVC", 50.0, 0.3, c) for c in conc]) \
                + _blank_counts(r2, adc)
            hit = np.where(sg.max(axis=1) >= top)[0]
            sat_c.append(float(conc[hit[0]]) if len(hit) else None)
        out[name] = {"values": vals.round(4).tolist(),
                     "slope_S1_counts_per_mg_L": sl[:, 0].round(4).tolist(),
                     "slope_S2_counts_per_mg_L": sl[:, 1].round(4).tolist(),
                     "saturation_conc_mg_L": sat_c}
        span = float(np.nanmax(sl[:, 0]) / max(np.nanmin(sl[:, 0]), 1e-12))
        rank[name] = span
        ax.semilogy(vals, np.maximum(sl[:, 0], 1e-6), "o-", label="S1")
        ax.semilogy(vals, np.maximum(sl[:, 1], 1e-6), "s-", label="S2")
        ax.set_xlabel(name)
        if rng_[0] > 0 and rng_[1] / rng_[0] > 20:
            ax.set_xscale("log")
    axes[0].set_ylabel("counts per mg/L")
    axes[0].legend(fontsize=8)
    fig.suptitle("Low-concentration sensitivity across the documented ranges of each rig parameter")
    fig.tight_layout()
    fig.savefig(out_dir() / "figures" / "geometry_sensitivity.png", dpi=120)
    plt.close(fig)
    ordered = dict(sorted(rank.items(), key=lambda kv: -kv[1]))
    return {"baseline_slope_counts_per_mg_L": base_slope.round(4).tolist(), "sweeps": out,
            "span_of_S1_slope_max_over_min": ordered, "most_sensitive": list(ordered)[:3]}


def size_confound(cfg, rig) -> dict:
    adc = G.adc_settings(cfg)
    res = {}
    for pol in ("PVC", "NYLON"):
        s20 = signal_counts(cfg, rig, adc, pol, 20.0, 0.3, 1.0)
        s80 = signal_counts(cfg, rig, adc, pol, 80.0, 0.3, 1.0)
        res[pol] = {"counts_20um": s20.round(3).tolist(), "counts_80um": s80.round(3).tolist(),
                    "ratio_80_over_20": (s80 / np.maximum(s20, 1e-9)).round(3).tolist()}
    return {"concentration_mg_L": 1.0, "median_size_um": [20, 80], "by_polymer": res,
            "note": "mass concentration fixed; signal per mg/L depends on particle size"}


def detection_limits(cfg, rig, n_blank=400, seed=21) -> dict:
    out = {}
    for model in cfg["adc"]["options"]:
        adc = G.adc_settings(cfg, model)
        rng = np.random.default_rng(seed)
        blank = np.array([det.readings(np.zeros(4), rig, 0.0, adc, cfg["noise"],
                                       cfg["laser"]["drift_pct"], cfg["medium"]["temperature_K"],
                                       rng)["S"] for _ in range(n_blank)], float)
        low_c = 0.1
        r = _power(cfg, rig, "PVC", 50.0, 0.3, low_c)
        low = np.array([det.readings(r["P_W"], rig, r["n_per_m3"], adc, cfg["noise"],
                                     cfg["laser"]["drift_pct"], cfg["medium"]["temperature_K"],
                                     rng)["S"] for _ in range(n_blank)], float)
        slope = _slope(cfg, rig, adc)
        sb, sl = blank.std(0, ddof=1), low.std(0, ddof=1)
        out[model] = {
            "slope_counts_per_mg_L": slope.round(4).tolist(),
            "blank_std_counts": sb.round(4).tolist(),
            "std_at_0.1_mg_L_counts": sl.round(4).tolist(),
            "LOD_mg_L_blank_sigma": (3.3 * sb / slope).round(5).tolist(),
            "LOQ_mg_L_blank_sigma": (10 * sb / slope).round(5).tolist(),
            "LOD_mg_L_low_conc_sigma": (3.3 * sl / slope).round(5).tolist(),
            "LOQ_mg_L_low_conc_sigma": (10 * sl / slope).round(5).tolist(),
            "adc_lsb_in_mg_L_S1": float(1.0 / slope[0]),
        }
    return {"detector_index_order": ["S1", "S2", "S3", "S4"], "adc": out,
            "note": "depends on the rig parameters in configs/scattering.yaml"}


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or G.load_scatter_config()
    rig = O.rig_from_config(cfg)
    res = {
        "meta": run_meta(),
        "config_sha256": G.config_sha256(cfg),
        "angular_curves": angular_curves(cfg, rig),
        "polymer_contrast": polymer_contrast(cfg, rig),
        "linearity": linearity(cfg, rig),
        "geometry_sensitivity": geometry_sensitivity(cfg, rig),
        "size_confound": size_confound(cfg, rig),
        "detection_limits": detection_limits(cfg, rig),
    }
    (out_dir() / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf8")
    return res
