"""Generate the multi-angle scattering table from the configured rig model."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from mp.config import repo_root, resolve
from mp.scatter import detector as det
from mp.scatter import optics as O

OUT_NAME = "scattering_v1.csv"


def load_scatter_config(path=None) -> dict:
    p = path or (repo_root() / "configs" / "scattering.yaml")
    return yaml.safe_load(open(p, encoding="utf8"))


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def concentration_levels(cfg: dict) -> np.ndarray:
    c = cfg["concentration"]
    return np.geomspace(c["min"], c["max"], c["levels"])


def geometries(cfg: dict, base: O.Rig, rng: np.random.Generator) -> list[tuple[str, O.Rig]]:
    """Default geometry plus perturbed copies drawn from the documented ranges."""
    out = [("g0", base)]
    j = cfg["geometry"]["jitter"]
    for k in range(cfg["geometry"].get("samples", 0)):
        th = tuple(float(np.clip(t + rng.uniform(-j["theta_deg"], j["theta_deg"]), 5, 175))
                   for t in base.theta_deg)
        out.append((f"g{k + 1}", replace(
            base, theta_deg=th,
            distance_m=base.distance_m * (1 + rng.uniform(-j["distance_rel"], j["distance_rel"])),
            gain_scale=1 + rng.uniform(-j["gain_rel"], j["gain_rel"]))))
    return out


def adc_settings(cfg: dict, model: str | None = None) -> dict:
    model = model or cfg["adc"]["model"]
    return {"model": model, **cfg["adc"]["options"][model]}


def expected_signal(cfg: dict, rig: O.Rig, polymer: str, d_med: float, sigma: float,
                    conc: float, mode: str, popt=None, nodes=None, seed: int = 0) -> dict:
    """Mean optical power per detector for one condition."""
    p = cfg["polymers"][polymer]
    nodes = nodes if nodes is not None else O.lognormal_nodes(d_med, sigma, cfg["size_distribution"]["nodes"])
    popt = popt or O.particle_optics(p["n"], p["k"], nodes, rig, with_phase=(mode == "monte_carlo"))
    return O.detector_power(conc, popt, rig, p["density_g_cm3"], nodes, mode,
                            mc_cfg=cfg.get("monte_carlo"), seed=seed)


def generate(cfg: dict | None = None, mode: str | None = None, adc_model: str | None = None,
             replicates: int | None = None, progress: bool = True) -> pd.DataFrame:
    cfg = cfg or load_scatter_config()
    mode = mode or cfg["multiple_scattering"]
    reps = replicates or cfg["replicates"]
    rng = np.random.default_rng(cfg["seed"])
    base = O.rig_from_config(cfg)
    adc = adc_settings(cfg, adc_model)
    sha = config_sha256(cfg)
    levels = concentration_levels(cfg)
    geos = geometries(cfg, base, rng)
    rows = []
    sid = 0

    def emit(rig, gid, polymer, dmed, sig, conc, rep, P, tau, n_dens):
        nonlocal sid
        out = det.readings(P, rig, n_dens, adc, cfg["noise"], cfg["laser"]["drift_pct"],
                           cfg["medium"]["temperature_K"], rng)
        row = {"sample_id": sid, "source": cfg["source_label"], "polymer": polymer,
               "conc_mg_L": float(conc), "d_median_um": dmed, "sigma_ln": sig,
               "replicate": rep, "geometry_id": gid, "adc_model": adc["model"]}
        for i, t in enumerate(rig.theta_deg, 1):
            row[f"theta_{i}"] = t
        for i in range(4):
            row[f"P{i + 1}_W"] = out["P_W"][i]
        for i in range(4):
            row[f"V{i + 1}"] = out["V"][i]
        for i in range(4):
            row[f"S{i + 1}"] = int(out["S"][i])
        row.update({"tau": tau, "saturated_any": bool(out["saturated"].any()),
                    "config_sha256": sha, "seed": cfg["seed"]})
        rows.append(row)
        sid += 1

    work = [(gid, rig, pol, opt) for gid, rig in geos for pol in cfg["polymers"]
            for opt in cfg["size_distribution"]["options"]]
    it = tqdm(work, desc="scattering", unit="cond") if progress else work
    for gid, rig, pol, opt in it:
        nodes = O.lognormal_nodes(opt["d_median_um"], opt["sigma_ln"],
                                  cfg["size_distribution"]["nodes"])
        popt = O.particle_optics(cfg["polymers"][pol]["n"], cfg["polymers"][pol]["k"],
                                 nodes, rig, with_phase=(mode == "monte_carlo"))
        for li, conc in enumerate(levels):
            r = O.detector_power(conc, popt, rig, cfg["polymers"][pol]["density_g_cm3"], nodes,
                                 mode, mc_cfg=cfg.get("monte_carlo"), seed=cfg["seed"] + li)
            for rep in range(reps):
                emit(rig, gid, pol, opt["d_median_um"], opt["sigma_ln"], conc, rep,
                     r["P_W"], r["tau"], r["n_per_m3"])
    if cfg["concentration"].get("blanks", True):
        for gid, rig in geos:
            for rep in range(2 * reps):
                emit(rig, gid, "blank", 0.0, 0.0, 0.0, rep, np.zeros(4), 0.0, 0.0)
    return pd.DataFrame(rows)


def output_path(cfg: dict, name: str = OUT_NAME):
    return resolve("data/processed") / name


def run(cfg: dict | None = None, adc_model: str | None = None) -> pd.DataFrame:
    cfg = cfg or load_scatter_config()
    df = generate(cfg, adc_model=adc_model)
    name = OUT_NAME if adc_model in (None, cfg["adc"]["model"]) else f"scattering_v1_{adc_model}.csv"
    out = output_path(cfg, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    yaml.safe_dump(copy.deepcopy(cfg), open(out.with_suffix(".config.yaml"), "w", encoding="utf8"),
                   sort_keys=False)
    print(f"{out.name}: {len(df)} rows, config {df['config_sha256'].iloc[0][:12]}")
    return df
