"""results/phase3_summary.md, generated from the analysis and regression JSON files."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mp.config import resolve  # noqa: E402


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def regression_figure(reg: dict, path) -> None:
    schemes = reg["E1"]["schemes"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(5.5 * len(schemes), 3.8), sharey=True)
    for ax, (sch, res) in zip(np.atleast_1d(axes), schemes.items()):
        names = list(res)
        r2 = np.array([res[n]["r2"] for n in names])
        lo = np.array([res[n]["ci95"]["r2"][0] for n in names])
        hi = np.array([res[n]["ci95"]["r2"][1] for n in names])
        ax.bar(names, r2, yerr=[np.clip(r2 - lo, 0, None), np.clip(hi - r2, 0, None)],
               color="#4c72b0", capsize=3)
        ax.set_title(sch.replace("_", " "))
        ax.set_ylabel("R² (95 % CI)")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
    fig.suptitle("Concentration regression, fixed rig geometry, unseen groups")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary() -> str:
    res = resolve("results")
    an = json.loads((res / "sensing" / "analysis.json").read_text(encoding="utf8"))
    reg = json.loads((res / "phase3_regression.json").read_text(encoding="utf8"))
    regression_figure(reg, res / "sensing" / "figures" / "regression_e1.png")
    L = ["## Multi-angle scattering: model results (generated)", ""]
    L += [f"Config hash `{an['config_sha256'][:12]}`; rig parameters are not measured, see "
          "`configs/scattering.yaml`.", ""]

    L += ["### Concentration regression, fixed geometry (E1)", ""]
    for sch, rows in reg["E1"]["schemes"].items():
        L += [f"**{sch.replace('_', ' ')}** ({reg['E1']['n_rows']} rows, "
              f"ADC {reg['E1']['adc_model']}):", "",
              "| Model | R² (95 % CI) | RMSE mg/L | MAE mg/L | MAPE % | 90 % interval coverage |",
              "|---|---|---|---|---|---|"]
        for n, v in rows.items():
            ci = v["ci95"]["r2"]
            L.append(f"| {n} | {v['r2']:.3f} ({ci[0]:.3f} to {ci[1]:.3f}) | {v['rmse']:.3f} | "
                     f"{v['mae']:.3f} | {v['mape_pct']:.1f} | "
                     f"{v['conformal']['empirical_coverage']:.2f} |")
        L.append("")
    L += ["![E1 regression](sensing/figures/regression_e1.png)", ""]

    rob = reg["robustness"]
    L += [f"### Robustness: train on {rob['train_geometry']}, test on perturbed geometries", "",
          "R² of log10(concentration), re-baselined per geometry (blank level measured on each "
          "geometry); the linear-scale R² and CIs are in `results/phase3_regression.json`.", "",
          "| Model | " + " | ".join(next(iter(rob["variants"].values()))
                                    ["random_forest"].keys()) + " |",
          "|---|" + "---|" * len(next(iter(rob["variants"].values()))["random_forest"])]
    for n, per in rob["variants"]["rebaselined_per_geometry"].items():
        L.append(f"| {n} | " + " | ".join(f"{v['log10_scale']['r2']:.2f}" for v in per.values()) + " |")
    L += ["", "Same, but the blank level of the training geometry is applied to every geometry:", "",
          "| Model | all_others |", "|---|---|"]
    for n, per in rob["variants"]["baseline_from_train_geometry"].items():
        L.append(f"| {n} | {per['all_others']['log10_scale']['r2']:.2f} |")
    L.append("")

    pc = an["polymer_contrast"]
    L += ["### PVC vs Nylon from scattering alone", "",
          f"d′ = |mean difference| / pooled noise, {pc['n_readings']} readings per cell, "
          f"{pc['size']}.", "", "| Conc. mg/L | d′ S1 | d′ S2 | d′ S3 | d′ S4 | rel. diff S1 |",
          "|---|---|---|---|---|---|"]
    for c, v in pc["by_concentration_mg_L"].items():
        s = v["snr_d_prime"]
        L.append(f"| {c} | {s[0]:.2f} | {s[1]:.2f} | {s[2]:.2f} | {s[3]:.2f} | "
                 f"{v['relative_difference'][0]:+.3f} |")
    L += ["", "Differences at equal mass concentration come mostly from the density difference "
          "(different particle numbers per mg/L), not from the refractive index.", ""]

    lin = an["linearity"]
    L += ["### Linearity and dynamic range", "",
          f"Optical depth reaches 0.1 at {lin['conc_at_tau_0.1_mg_L']:.0f} mg/L. "
          f"Saturation (10-bit ADC): {lin['saturation_conc_mg_L']}.", ""]

    dl = an["detection_limits"]["adc"]
    L += ["### Detection limits (S1, S2)", "",
          "| ADC | slope S1 counts per mg/L | LOD S1 mg/L | LOQ S1 mg/L | LOD S2 mg/L | LOQ S2 mg/L |",
          "|---|---|---|---|---|---|"]
    for m, v in dl.items():
        L.append(f"| {m} | {v['slope_counts_per_mg_L'][0]:.2f} | "
                 f"{v['LOD_mg_L_low_conc_sigma'][0]:.3f} | {v['LOQ_mg_L_low_conc_sigma'][0]:.3f} | "
                 f"{v['LOD_mg_L_low_conc_sigma'][1]:.3f} | {v['LOQ_mg_L_low_conc_sigma'][1]:.3f} |")
    L += ["", "LOD = 3.3 σ / slope and LOQ = 10 σ / slope, with σ the reading noise at 0.1 mg/L "
          "(particle-number noise included). Blank-only σ is below one count for the 10-bit "
          "ADC, which makes a blank-based LOD meaningless there (`analysis.json`).", ""]

    gs = an["geometry_sensitivity"]
    L += ["### Geometry sensitivity", "",
          "Span (max / min) of the S1 slope across the documented range of each parameter:", "",
          "| Parameter | Span |", "|---|---|"]
    for k, v in gs["span_of_S1_slope_max_over_min"].items():
        L.append(f"| {k} | {v:.1f} |")
    L.append("")

    sc = an["size_confound"]["by_polymer"]["PVC"]
    L += ["### Size dependence at fixed mass concentration", "",
          f"PVC at 1 mg/L, median size 20 µm to 80 µm: signal ratio per detector "
          f"{sc['ratio_80_over_20']}.", ""]
    out = res / "phase3_summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf8")
    return str(out)
