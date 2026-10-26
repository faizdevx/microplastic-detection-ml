"""Unified analysis: polymer from an image, concentration from four detector counts."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from mp import __version__
from mp.config import load_config, load_named, repo_root, resolve
from mp.eval.metrics import run_meta

SIZE_BINS_PX = [0, 10, 20, 30, 50, 80, 120, 200, 100000]
MIN_PARTICLES = 3
ESTIMATE_NOTE = "rig model estimate, not a measurement"
ECE_NOTE = "probabilities are not well calibrated (hold-out ECE {ece:.3f})"


# ------------------------------------------------------------------ polymer -----------
def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_cache_path() -> Path:
    return repo_root() / "data" / "models" / "baseline_a.joblib"


def _fit_polymer_model(cfg: dict, ccfg: dict, cv: dict, force: bool = False) -> dict:
    """Fit the selected baseline-A model on all CV images; cache with provenance."""
    from mp.data.dataset import load_pool
    from mp.imaging.features import feature_columns, particles_path
    from mp.models import classifier as C

    sel = cv["selected"]
    geometry = sel["geometry"]
    ppath = particles_path(cfg, geometry)
    stamp = {
        "particles_sha256": _sha256_file(ppath),
        "splits_sha256": _sha256_file(resolve(cfg["paths"]["splits"])),
        "model": sel["model"], "params": sel["params"], "geometry": geometry,
        "mp_version": __version__,
    }
    cache = model_cache_path()
    if cache.exists() and not force:
        obj = joblib.load(cache)
        if obj["stamp"] == stamp:
            return obj
    pool = load_pool(cfg)
    P = C.load_particles(geometry, cfg, pool)
    P = P[P["rel_path"].isin(pool.cv()["rel_path"])]
    feats = feature_columns(ccfg["feature_groups"])
    model = C.make_model(sel["model"], sel["params"], ccfg["seed"])
    model.fit(P[feats], P["y"], clf__sample_weight=C.particle_weights(P))
    cvimgs = pool.cv()
    stats = pd.read_csv(resolve(cfg["paths"]["image_stats"]))
    st = cvimgs.merge(stats, on="rel_path")
    bright = st[["mean_R", "mean_G", "mean_B"]].mean(axis=1)
    obj = {
        "stamp": stamp, "model": model, "features": feats,
        "brightness_range": [float(bright.min()), float(bright.max())],
        "resolutions": sorted({(int(w), int(h)) for w, h in zip(cvimgs["width"], cvimgs["height"])}),
        "git_commit": run_meta()["git_commit"],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, cache)
    return obj


@lru_cache(maxsize=1)
def _polymer_bundle() -> dict:
    cfg, ccfg = load_config(), load_named("classifier")
    cv = json.loads((resolve(cfg["paths"]["results"]) / "phase2_cv.json").read_text(encoding="utf8"))
    return {"cfg": cfg, "ccfg": ccfg, "fit": _fit_polymer_model(cfg, ccfg, cv),
            "ece": json.loads((resolve(cfg["paths"]["results"]) / "phase2_holdout.json")
                              .read_text(encoding="utf8"))["metrics"]["ece"]}


def _size_hist(diam_px: np.ndarray) -> dict:
    counts, edges = np.histogram(diam_px, bins=SIZE_BINS_PX)
    return {"bin_edges_px": [float(e) for e in edges[:-1]] + ["inf"],
            "counts": [int(c) for c in counts]}


def analyse_image(path: str | Path, with_overlay: bool = False) -> dict:
    """Polymer decision and particle statistics for one fluorescence image."""
    from mp.imaging import features as F
    from mp.imaging import normalize as nz
    from mp.imaging import segment as sg

    b = _polymer_bundle()
    fit, ccfg = b["fit"], b["ccfg"]
    warnings_: list[str] = []
    with Image.open(path) as im:
        w, h = im.size
    rgb = nz.load_rgb(path)
    brightness = float(rgb.reshape(-1, 3).mean())
    a = F.prepare(path, "resample", ccfg["preprocess"])
    labels = sg.segment(a["bgsub"], ccfg["segmentation"])
    parts = F.particle_features(labels, a["bgsub"], a["norm"])
    n = len(parts)
    reason = None
    if abs(w / h - 4 / 3) > 0.01:
        reason = f"image is {w}x{h}, not 4:3"
    elif (w, h) not in fit["resolutions"]:
        reason = f"resolution {w}x{h} was not seen in training"
    elif not fit["brightness_range"][0] <= brightness <= fit["brightness_range"][1]:
        reason = (f"mean brightness {brightness:.1f} is outside the training range "
                  f"{fit['brightness_range'][0]:.1f}-{fit['brightness_range'][1]:.1f}")
    elif n < MIN_PARTICLES:
        reason = f"only {n} particles found (need at least {MIN_PARTICLES})"
    poly = {"label": None, "p_nylon": None, "confidence": None,
            "confidence_note": ECE_NOTE.format(ece=b["ece"])}
    if reason:
        poly["reason"] = reason
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(fit["model"].predict_proba(parts[fit["features"]])[:, 1].mean())
        poly.update(label="NYLON" if p >= 0.5 else "PVC", p_nylon=p, confidence=max(p, 1 - p))
        warnings_.append("class is partly predictable from background and exposure alone; "
                         "see MODEL_CARD.md")
    um = load_config().get("um_per_px")
    hist = _size_hist(parts["eq_diameter_px"].to_numpy()) if n else _size_hist(np.array([]))
    out = {
        "polymer": poly,
        "particles": {"count": int(n), "size_hist_px": hist,
                      "um_per_px": um, "pixel_size_note": None if um else "pixel size not calibrated"},
        "concentration": None,
        "warnings": warnings_,
    }
    if with_overlay:
        out["_overlay"] = (a["geo"], labels)
    return out


# ------------------------------------------------------------- concentration ----------
def select_regression(write: bool = True) -> dict:
    """Best model by E1 R² (concentration-level CV); saved to results/phase3_selected.json."""
    res = resolve("results")
    reg = json.loads((res / "phase3_regression.json").read_text(encoding="utf8"))
    rows = reg["E1"]["schemes"]["by_concentration_level"]
    best = max(rows, key=lambda n: rows[n]["r2"])
    sel = {"model": best, "geometry": reg["E1"]["geometry"], "adc_model": reg["E1"]["adc_model"],
           "e1_r2": rows[best]["r2"], "e1_r2_ci95": rows[best]["ci95"]["r2"],
           "rule": "highest R2 in E1, concentration-level grouped CV"}
    if write:
        (res / "phase3_selected.json").write_text(json.dumps(sel, indent=2), encoding="utf8")
    return sel


def _scatter_table(adc: str) -> pd.DataFrame:
    base = resolve("data/processed")
    from mp.scatter.generate import load_scatter_config

    default = load_scatter_config()["adc"]["model"]
    path = base / ("scattering_v1.csv" if adc == default else f"scattering_v1_{adc}.csv")
    if not path.exists():
        hint = "make sim" if adc == default else f"python -m mp.cli scatter-generate --adc {adc}"
        raise FileNotFoundError(f"{path.name} not found; run `{hint}`")
    return pd.read_csv(path)


@lru_cache(maxsize=4)
def _conc_bundle(adc: str) -> dict:
    from mp.models import regression as R

    rcfg = load_named("regression")
    res = resolve("results")
    sel_path = res / "phase3_selected.json"
    sel = json.loads(sel_path.read_text(encoding="utf8")) if sel_path.exists() else select_regression()
    df = _scatter_table(adc)
    d = R.dark_subtract(df)
    g = sel["geometry"]
    blank = df[(df["polymer"] == "blank") & (df["geometry_id"] == g)]
    blank_mean = blank[R.S_COLS].mean().to_numpy()
    tr = R.training_rows(d[(d["geometry_id"] == g) & (d["adc_model"] == adc)], rcfg)
    X, y = R.make_features(tr), np.log10(tr["conc_mg_L"].to_numpy())
    model = R.make_model(sel["model"], rcfg, adc).fit(X, y)
    levels = tr["conc_mg_L"].round(6).to_numpy()
    a, b = R._split_by_group(levels, rcfg["conformal"]["calibration_fraction"], rcfg["seed"])
    m2 = R.make_model(sel["model"], rcfg, adc).fit(X[a], y[a])
    q = R.conformal_q(np.abs(y[b] - m2.predict(X[b])), rcfg["conformal"]["coverage"])
    span = (X.max() - X.min()).to_numpy()
    top = {"mcp3008": 2**10 - 1, "ads1115": 2**15 - 1}.get(adc, 2**10 - 1)
    return {"R": R, "sel": sel, "model": model, "q": q, "blank": blank_mean,
            "lo": X.min().to_numpy() - 0.05 * span, "hi": X.max().to_numpy() + 0.05 * span,
            "cols": list(X.columns), "top": top, "adc": adc, "conc_range": [float(tr["conc_mg_L"].min()),
                                                                           float(tr["conc_mg_L"].max())]}


def estimate_concentration(sensors, adc: str = "mcp3008") -> dict:
    """Concentration estimate and 90 % interval, or a refusal with a reason."""
    b = _conc_bundle(adc)
    R = b["R"]
    s = np.asarray(sensors, float)
    if s.shape != (4,):
        raise ValueError("need exactly four sensor values S1 S2 S3 S4")
    out = {"mg_L": None, "interval_90": None, "note": ESTIMATE_NOTE,
           "model": b["sel"]["model"], "adc": adc, "geometry": b["sel"]["geometry"]}
    if np.any(s >= b["top"]):
        out["reason"] = f"a detector is saturated (>= {b['top']} counts)"
        return out
    ds = s - b["blank"]
    if ds[0] < 0 and ds[1] < 0:
        out["reason"] = "no signal above the blank level on S1 and S2"
        return out
    row = pd.DataFrame([{**{f"S{i + 1}": s[i] for i in range(4)},
                         **{f"dS{i + 1}": ds[i] for i in range(4)}}])
    X = R.make_features(row)[b["cols"]]
    x = X.to_numpy()[0]
    if np.any(x < b["lo"]) or np.any(x > b["hi"]):
        bad = [c for c, v, lo, hi in zip(b["cols"], x, b["lo"], b["hi"]) if v < lo or v > hi]
        out["reason"] = f"features outside the training range: {', '.join(bad)}"
        return out
    pred = float(b["model"].predict(X)[0])
    out["mg_L"] = float(10**pred)
    out["interval_90"] = [float(10 ** (pred - b["q"])), float(10 ** (pred + b["q"]))]
    out["training_range_mg_L"] = b["conc_range"]
    return out


def calibration_curve(adc: str = "mcp3008", n: int = 40) -> dict:
    """Model-predicted counts above blank versus concentration (for plots)."""
    from mp.scatter import analysis as A
    from mp.scatter import generate as G
    from mp.scatter import optics as O

    cfg = G.load_scatter_config()
    rig = O.rig_from_config(cfg)
    adc_s = G.adc_settings(cfg, adc)
    conc = np.geomspace(0.1, 10.0, n)
    d = cfg["size_distribution"]["options"][1]
    sig = np.array([A.signal_counts(cfg, rig, adc_s, "PVC", d["d_median_um"], d["sigma_ln"], c)
                    for c in conc])
    return {"conc_mg_L": conc.tolist(), "counts_above_blank": sig.tolist(),
            "blank_counts": A._blank_counts(rig, adc_s).tolist()}


# ---------------------------------------------------------------------- main ----------
def versions() -> dict:
    import sklearn

    return {"mp": __version__, "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform()}


def analyse(image: str | Path | None = None, sensors=None, adc: str = "mcp3008") -> dict:
    if image is None and sensors is None:
        raise ValueError("give an image, sensor values, or both")
    out = (analyse_image(image) if image is not None else
           {"polymer": None, "particles": None, "concentration": None, "warnings": []})
    if sensors is not None:
        out["concentration"] = estimate_concentration(sensors, adc)
    out["versions"] = versions()
    out["git_commit"] = run_meta()["git_commit"]
    return out


def format_text(r: dict) -> str:
    L = []
    p = r.get("polymer")
    if p:
        if p["label"]:
            L.append(f"polymer: {p['label']} (P(NYLON) {p['p_nylon']:.3f}, confidence "
                     f"{p['confidence']:.3f}); {p['confidence_note']}")
        else:
            L.append(f"polymer: cannot decide ({p.get('reason', 'unknown')})")
        pc = r["particles"]
        L.append(f"particles: {pc['count']}; size histogram (px) {pc['size_hist_px']['counts']}")
    c = r.get("concentration")
    if c:
        if c["mg_L"] is None:
            L.append(f"concentration: refused ({c['reason']})")
        else:
            lo, hi = c["interval_90"]
            L.append(f"concentration: {c['mg_L']:.2f} mg/L (90 % interval {lo:.2f} to {hi:.2f}); "
                     f"{c['note']}")
    L += [f"warning: {w}" for w in r.get("warnings", [])]
    return "\n".join(L)
