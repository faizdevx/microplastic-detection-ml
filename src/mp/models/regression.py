"""Concentration regression from the four detector counts.

Works on any table with the columns of data/processed/scattering_v1.csv. Blank rows
(polymer == "blank") give the dark/offset level of each geometry and are subtracted.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mp.config import load_named, resolve
from mp.eval.metrics import run_meta

S_COLS = ["S1", "S2", "S3", "S4"]
DS_COLS = [f"dS{i}" for i in range(1, 5)]
EPS = 1.0
MODELS = ("lasso", "ridge", "pls", "random_forest", "svr_rbf", "gaussian_process",
          "physics_inversion")


# --------------------------------------------------------------------- data -----------
def load_table(path=None, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_named("regression")
    return pd.read_csv(resolve(path or cfg["input"]))


def dark_subtract(df: pd.DataFrame, baseline_geometry: str | None = None) -> pd.DataFrame:
    """dS_i = S_i - mean blank S_i of the same geometry (or of `baseline_geometry`)."""
    blank = df[df["polymer"] == "blank"]
    if blank.empty:
        raise ValueError("no blank rows: cannot dark-subtract")
    key = ["geometry_id", "adc_model"]
    base = blank.groupby(key)[S_COLS].mean()
    out = df.copy()
    for _, row in out[key].drop_duplicates().iterrows():
        gid, adc = row["geometry_id"], row["adc_model"]
        ref = (baseline_geometry or gid, adc)
        m = (out["geometry_id"] == gid) & (out["adc_model"] == adc)
        for i, c in enumerate(S_COLS):
            out.loc[m, DS_COLS[i]] = out.loc[m, c] - base.loc[ref, c]
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    ds = df[DS_COLS].to_numpy(float)
    pos = np.clip(ds, 0, None)
    f = pd.DataFrame(index=df.index)
    for i in range(4):
        f[f"dS{i + 1}"] = ds[:, i]
        f[f"logS{i + 1}"] = np.log1p(pos[:, i])
    f["S_mean"] = ds.mean(1)
    f["S_std"] = ds.std(1)
    f["r14"] = (pos[:, 0] + EPS) / (pos[:, 3] + EPS)
    f["r24"] = (pos[:, 1] + EPS) / (pos[:, 3] + EPS)
    f["r13"] = (pos[:, 0] + EPS) / (pos[:, 2] + EPS)
    f["fwd_back"] = (pos[:, 0] + pos[:, 1] + EPS) / (pos[:, 2] + pos[:, 3] + EPS)
    return f


def training_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df[df["polymer"] != "blank"] if cfg["exclude_blank_from_training"] else df
    if cfg["drop_saturated"] and "saturated_any" in d:
        d = d[~d["saturated_any"].astype(bool)]
    return d[d["conc_mg_L"] > 0]


# -------------------------------------------------------------------- models ----------
class Sk:
    """sklearn estimator on the engineered features; target is log10(conc)."""

    def __init__(self, name: str, params: dict, seed: int):
        self.name, self.params, self.seed = name, params, seed

    def _build(self):
        p = self.params
        if self.name == "lasso":
            return make_pipeline(StandardScaler(), Lasso(alpha=p["alpha"], max_iter=20000))
        if self.name == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
        if self.name == "pls":
            return make_pipeline(StandardScaler(), PLSRegression(n_components=p["n_components"]))
        if self.name == "random_forest":
            return RandomForestRegressor(n_estimators=p["n_estimators"],
                                         min_samples_leaf=p["min_samples_leaf"],
                                         random_state=self.seed, n_jobs=-1)
        if self.name == "svr_rbf":
            return make_pipeline(StandardScaler(), SVR(C=p["C"], epsilon=p["epsilon"],
                                                       gamma=p["gamma"]))
        if self.name == "gaussian_process":
            k = ConstantKernel(1.0) * RBF(length_scale=np.ones(1)) + WhiteKernel(0.1)
            return make_pipeline(StandardScaler(),
                                 GaussianProcessRegressor(k, normalize_y=True,
                                                          random_state=self.seed))
        raise ValueError(self.name)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if self.name == "gaussian_process" and len(X) > self.params["max_train"]:
            idx = np.random.default_rng(self.seed).choice(len(X), self.params["max_train"],
                                                          replace=False)
            X, y = X.iloc[idx], y[idx]
        self.m = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(X.to_numpy(), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.m.predict(X.to_numpy())).ravel()


class PhysicsInversion:
    """Invert the forward model of the rig for the concentration.

    A table of expected counts above blank versus concentration is computed for the
    nominal geometry (both polymers averaged, nominal size). A global gain is fitted on
    the training data, detector weights come from the training residuals, and each
    sample is inverted by a 1-D search over the concentration grid.
    """

    def __init__(self, params: dict, scatter_cfg: dict | None = None, adc_model: str = "mcp3008"):
        from mp.scatter import analysis as A
        from mp.scatter import generate as G
        from mp.scatter import optics as O

        self.params = params
        self.cfg = scatter_cfg or G.load_scatter_config()
        rig = O.rig_from_config(self.cfg)
        adc = G.adc_settings(self.cfg, adc_model)
        self.grid = np.geomspace(0.02, 100.0, 400)
        d = self.cfg["size_distribution"]["options"][1]
        tab = []
        for pol in self.cfg["polymers"]:
            tab.append([A.signal_counts(self.cfg, rig, adc, pol, d["d_median_um"],
                                        d["sigma_ln"], c) for c in self.grid])
        self.table = np.mean(tab, axis=0)  # (grid, 4)
        self.gain, self.w = 1.0, np.ones(4)

    def fit(self, X: pd.DataFrame, y_log: np.ndarray):
        ds = np.clip(X[DS_COLS].to_numpy(float), 0, None)
        c = 10**y_log
        exp = np.stack([np.interp(c, self.grid, self.table[:, i]) for i in range(4)], 1)
        if self.params.get("fit_gain", True):
            ok = exp[:, 0] > 5
            self.gain = float(np.median(ds[ok, 0] / exp[ok, 0])) if ok.any() else 1.0
        r = np.log1p(ds) - np.log1p(self.gain * exp)
        self.w = 1.0 / np.maximum(r.var(0), 1e-3)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ds = np.clip(X[DS_COLS].to_numpy(float), 0, None)
        model = np.log1p(self.gain * self.table)                    # (grid, 4)
        obs = np.log1p(ds)                                          # (n, 4)
        cost = (((obs[:, None, :] - model[None]) ** 2) * self.w).sum(2)
        return np.log10(self.grid[np.argmin(cost, axis=1)])


def make_model(name: str, cfg: dict, adc_model: str):
    if name == "physics_inversion":
        return PhysicsInversion(cfg["models"][name], adc_model=adc_model)
    return Sk(name, cfg["models"][name], cfg["seed"])


# ------------------------------------------------------------------- metrics ----------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res, ss_tot = float((err**2).sum()), float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "mape_pct": float(100 * np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12))),
    }


def bootstrap_ci(y_true, y_pred, groups, n=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    gids = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in gids}
    draws = {k: [] for k in ("r2", "rmse", "mae", "mape_pct")}
    for _ in range(n):
        pick = rng.choice(gids, size=len(gids), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        m = metrics(y_true[idx], y_pred[idx])
        for k in draws:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in draws.items()}


def summarise(y_true_c, y_pred_c, groups, cfg) -> dict:
    out = metrics(y_true_c, y_pred_c)
    out["ci95"] = bootstrap_ci(y_true_c, y_pred_c, groups, cfg["bootstrap"], cfg["seed"])
    out["n"] = int(len(y_true_c))
    return out


# --------------------------------------------------------------- conformal ------------
def conformal_q(resid_abs: np.ndarray, coverage: float) -> float:
    n = len(resid_abs)
    k = int(np.ceil((n + 1) * coverage))
    return float(np.sort(resid_abs)[min(k, n) - 1])


def _split_by_group(groups: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    cal = set(rng.choice(g, size=max(1, int(round(frac * len(g)))), replace=False))
    m = np.array([x in cal for x in groups])
    return ~m, m


# ----------------------------------------------------------------- experiments --------
def run_e1(df: pd.DataFrame, cfg: dict | None = None, geometry: str = "g0",
           adc_model: str | None = None) -> dict:
    """E1: fixed rig, unseen concentration levels (or unseen replicates)."""
    cfg = cfg or load_named("regression")
    d = dark_subtract(df)
    adc_model = adc_model or d["adc_model"].iloc[0]
    d = training_rows(d[(d["geometry_id"] == geometry) & (d["adc_model"] == adc_model)], cfg)
    X, y = make_features(d), np.log10(d["conc_mg_L"].to_numpy())
    out = {"geometry": geometry, "adc_model": adc_model, "n_rows": int(len(d)), "schemes": {}}
    for scheme in cfg["cv"]["schemes"]:
        groups = (d["conc_mg_L"].round(6).to_numpy() if scheme == "by_concentration_level"
                  else d["replicate"].to_numpy())
        gkf = GroupKFold(n_splits=min(cfg["cv"]["n_splits"], len(np.unique(groups))))
        res = {}
        for name in MODELS:
            pred = np.full(len(d), np.nan)
            cover, width = [], []
            for tr, te in gkf.split(X, y, groups):
                mdl = make_model(name, cfg, adc_model).fit(X.iloc[tr], y[tr])
                pred[te] = mdl.predict(X.iloc[te])
                # split conformal on the training part of the fold
                a, b = _split_by_group(groups[tr], cfg["conformal"]["calibration_fraction"], cfg["seed"])
                m2 = make_model(name, cfg, adc_model).fit(X.iloc[tr[a]], y[tr[a]])
                q = conformal_q(np.abs(y[tr[b]] - m2.predict(X.iloc[tr[b]])),
                                cfg["conformal"]["coverage"])
                p2 = m2.predict(X.iloc[te])
                cover.append(float(np.mean(np.abs(y[te] - p2) <= q)))
                width.append(2 * q)
            res[name] = {
                **summarise(10**y, 10**pred, groups, cfg),
                "log10_scale": metrics(y, pred),
                "conformal": {"target": cfg["conformal"]["coverage"],
                              "empirical_coverage": float(np.mean(cover)),
                              "mean_width_log10": float(np.mean(width))},
            }
        out["schemes"][scheme] = res
    return out


def run_robustness(df: pd.DataFrame, cfg: dict | None = None, adc_model: str | None = None) -> dict:
    """Train on one geometry, test on the others (perturbed angles, distance, gain)."""
    cfg = cfg or load_named("regression")
    train_g = cfg["robustness"]["train_geometry"]
    adc_model = adc_model or df["adc_model"].iloc[0]
    out = {"train_geometry": train_g, "adc_model": adc_model, "variants": {}}
    for variant, base in (("rebaselined_per_geometry", None), ("baseline_from_train_geometry", train_g)):
        d = dark_subtract(df, baseline_geometry=base)
        d = training_rows(d[d["adc_model"] == adc_model], cfg)
        X, y = make_features(d), np.log10(d["conc_mg_L"].to_numpy())
        tr_m = (d["geometry_id"] == train_g).to_numpy()
        levels = d["conc_mg_L"].round(6).to_numpy()
        res = {}
        for name in MODELS:
            mdl = make_model(name, cfg, adc_model).fit(X[tr_m], y[tr_m])
            a, b = _split_by_group(levels[tr_m], cfg["conformal"]["calibration_fraction"], cfg["seed"])
            idx = np.where(tr_m)[0]
            m2 = make_model(name, cfg, adc_model).fit(X.iloc[idx[a]], y[idx[a]])
            q = conformal_q(np.abs(y[idx[b]] - m2.predict(X.iloc[idx[b]])), cfg["conformal"]["coverage"])
            per = {}
            for gid in sorted(set(d["geometry_id"]) - {train_g}) + ["all_others"]:
                te = (~tr_m) if gid == "all_others" else (d["geometry_id"] == gid).to_numpy()
                pred = mdl.predict(X[te])
                per[gid] = {**summarise(10**y[te], 10**pred, levels[te], cfg),
                            "log10_scale": metrics(y[te], pred),
                            "conformal_coverage": float(np.mean(np.abs(y[te] - m2.predict(X[te])) <= q))}
            res[name] = per
        out["variants"][variant] = res
    return out


NEEDS_MEASURED = "needs a measured scattering table (source = measured)"


def _require_measured(df: pd.DataFrame) -> None:
    if "source" not in df or not (df["source"] == "measured").any():
        raise NotImplementedError(NEEDS_MEASURED)


def run_e2(model_df: pd.DataFrame, measured_df: pd.DataFrame, cfg=None):
    """E2: train on the model table, test on a measured table."""
    _require_measured(measured_df)
    raise NotImplementedError("E2 is not implemented yet")


def run_e3(measured_df: pd.DataFrame, cfg=None):
    """E3: train and test on measured data."""
    _require_measured(measured_df)
    raise NotImplementedError("E3 is not implemented yet")


def run_e4(model_df: pd.DataFrame, measured_df: pd.DataFrame, cfg=None):
    """E4: model table plus a few measured rows."""
    _require_measured(measured_df)
    raise NotImplementedError("E4 is not implemented yet")


def run(experiment: str = "all", cfg: dict | None = None) -> dict:
    cfg = cfg or load_named("regression")
    df = load_table(cfg=cfg)
    out = {"meta": run_meta(), "input": cfg["input"], "source_values": sorted(df["source"].unique())}
    if experiment in ("E1", "all"):
        out["E1"] = run_e1(df, cfg)
    if experiment in ("robustness", "all"):
        out["robustness"] = run_robustness(df, cfg)
    path = resolve("results") / "phase3_regression.json"
    old = json.loads(path.read_text(encoding="utf8")) if path.exists() and experiment != "all" else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=2), encoding="utf8")
    return old
