"""Confound controls for the polymer classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from mp.config import load_config, load_named, resolve
from mp.data.dataset import Pool, load_pool
from mp.imaging.features import BG_COLS, FEATURE_GROUPS, feature_columns, images_path
from mp.models import classifier as C

SIZE_COLS = ["width", "height", "file_size"]
STAT_COLS = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"]


def res_key(df: pd.DataFrame) -> pd.Series:
    return df["width"].astype(str) + "x" + df["height"].astype(str)


def acc_bal(y, prob) -> dict:
    ok = ~np.isnan(prob)
    pred = (prob[ok] >= 0.5).astype(int)
    return {
        "accuracy": float((pred == y[ok]).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y[ok], pred)),
        "n": int(ok.sum()),
        "n_pvc": int((y[ok] == 0).sum()),
        "n_nylon": int((y[ok] == 1).sum()),
    }


def reweight_particles(P: pd.DataFrame, y_col: str = "y") -> pd.DataFrame:
    """Recompute image weights on the rows present: 1/group size, equal class totals."""
    img = P.drop_duplicates("rel_path")[["rel_path", "group_id", y_col]].copy()
    img["w"] = 1.0 / img.groupby("group_id")["group_id"].transform("size")
    img["w"] = img["w"] / img.groupby(y_col)["w"].transform("sum")
    out = P.drop(columns=["weight"]).merge(
        img[["rel_path", "w"]].rename(columns={"w": "weight"}), on="rel_path"
    )
    return out


# ---------------------------------------------------------------- shortcut --------------
def _meta_table(pool: Pool, cfg: dict) -> pd.DataFrame:
    stats = pd.read_csv(resolve(cfg["paths"]["image_stats"]))
    return pool.images.merge(stats, on="rel_path")


def _shortcut_fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols, seed) -> np.ndarray:
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(tr[cols], tr["y"], sample_weight=tr["weight"])
    return rf.predict_proba(te[cols])[:, 1]


def shortcut_cv(meta: pd.DataFrame, pool: Pool, cols, seed, subset=None) -> dict:
    parts = []
    for k in range(pool.n_folds):
        cv = meta[meta["fold"] != -1]
        tr, te = cv[cv["fold"] != k], cv[cv["fold"] == k]
        if subset is not None:
            tr, te = tr[tr["rel_path"].isin(subset)], te[te["rel_path"].isin(subset)]
        te = te.copy()
        te["prob"] = _shortcut_fit_predict(tr, te, cols, seed)
        parts.append(te)
    oof = pd.concat(parts)
    return acc_bal(oof["y"].to_numpy(), oof["prob"].to_numpy())


def shortcut_cross(meta: pd.DataFrame, a: str, b: str, cols, seed) -> dict:
    cv = meta[meta["fold"] != -1]
    te = cv[res_key(cv) == b].copy()
    tr = cv[(res_key(cv) == a) & ~cv["group_id"].isin(te["group_id"])]
    te["prob"] = _shortcut_fit_predict(tr, te, cols, seed)
    return acc_bal(te["y"].to_numpy(), te["prob"].to_numpy())


# ---------------------------------------------------------------- controls --------------
def resolution_matched(P, pool, ccfg, name, params, feats, matched) -> dict:
    sub = pool.images[res_key(pool.images).isin(matched)]
    Psub = reweight_particles(P[P["rel_path"].isin(sub["rel_path"])])
    r = C.run_cv(name, Psub, pool, ccfg, feats, do_tune=False, params=params,
                 subset=pd.Index(sub["rel_path"]))
    return acc_bal(r["oof"]["y"].to_numpy(), r["oof"]["prob"].to_numpy())


def cross_resolution(P, pool, ccfg, name, params, feats, a, b) -> dict:
    cv = pool.cv()
    te = cv[res_key(cv) == b]
    tr = cv[(res_key(cv) == a) & ~cv["group_id"].isin(te["group_id"])]
    Ptr = reweight_particles(P[P["rel_path"].isin(tr["rel_path"])])
    P2 = pd.concat([Ptr, P[P["rel_path"].isin(te["rel_path"])]])
    img = C.eval_split(name, params, P2, tr, te, feats, ccfg["seed"])
    return acc_bal(img["y"].to_numpy(), img["prob"].to_numpy())


def background_only(pool: Pool, cfg: dict, geometry: str, ccfg: dict) -> dict:
    """Image-level classifier on background statistics only (particles masked out)."""
    bg = pd.read_csv(images_path(cfg, geometry))
    df = pool.images.merge(bg[["rel_path", *BG_COLS]], on="rel_path").dropna(subset=BG_COLS)
    spec = ccfg["models"][ccfg["controls"]["background_only_model"]]
    parts = []
    for k in range(pool.n_folds):
        cv = df[df["fold"] != -1]
        tr, te = cv[cv["fold"] != k], cv[cv["fold"] == k].copy()
        rf = RandomForestClassifier(
            n_estimators=spec.get("fixed", {}).get("n_estimators", 200),
            random_state=ccfg["seed"], n_jobs=-1,
        )
        rf.fit(tr[BG_COLS], tr["y"], sample_weight=tr["weight"])
        te["prob"] = rf.predict_proba(te[BG_COLS])[:, 1]
        parts.append(te)
    oof = pd.concat(parts)
    return acc_bal(oof["y"].to_numpy(), oof["prob"].to_numpy())


def _cv_fixed(P, imgs, name, params, feats, seed, n_folds) -> float:
    """Pooled image-level accuracy of a fixed-parameter model over the fold assignment."""
    parts = []
    for k in range(n_folds):
        tr, va = imgs[imgs["fold"] != k], imgs[imgs["fold"] == k]
        parts.append(C.eval_split(name, params, P, tr, va, feats, seed))
    oof = pd.concat(parts)
    ok = oof.dropna(subset=["prob"])
    return float(((ok["prob"] >= 0.5).astype(int) == ok["y"]).mean())


def _perm_once(seed_i, P, imgs, params, feats, seed, n_folds) -> float:
    rng = np.random.default_rng(seed_i)
    g = imgs.drop_duplicates("group_id")[["group_id", "y"]].reset_index(drop=True)
    g["y_perm"] = rng.permutation(g["y"].to_numpy())
    m = dict(zip(g["group_id"], g["y_perm"]))
    imgs2, P2 = imgs.copy(), P.copy()
    imgs2["y"] = imgs2["group_id"].map(m)
    P2["y"] = P2["group_id"].map(m)
    P2 = reweight_particles(P2)
    return _cv_fixed(P2, imgs2, "hist_gb", params, feats, seed, n_folds)


def permutation_test(P, pool, ccfg, feats, n_jobs: int = 4) -> dict:
    cc = ccfg["controls"]
    params = {"max_iter": cc["permutation_model_iter"]}
    imgs = pool.cv()
    seed, nf = ccfg["seed"], pool.n_folds
    obs = _cv_fixed(P, imgs, "hist_gb", params, feats, seed, nf)
    null = Parallel(n_jobs=n_jobs)(
        delayed(_perm_once)(1000 + i, P, imgs, params, feats, seed, nf)
        for i in range(cc["permutations"])
    )
    null = np.array(null)
    return {
        "model": "hist_gb (fixed, max_iter=%d)" % cc["permutation_model_iter"],
        "observed_accuracy": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_max": float(null.max()),
        "n_permutations": int(len(null)),
        "p_value": float((1 + (null >= obs).sum()) / (1 + len(null))),
    }


def feature_group_ablation(P, pool, ccfg, params_hgb=None) -> dict:
    out = {}
    for name_g, cols in {**{g: FEATURE_GROUPS[g] for g in FEATURE_GROUPS},
                         "all": feature_columns(list(FEATURE_GROUPS))}.items():
        r = C.run_cv("hist_gb", P, pool, ccfg, list(cols), do_tune=False,
                     params=params_hgb or {"max_iter": 100})
        out[name_g] = acc_bal(r["oof"]["y"].to_numpy(), r["oof"]["prob"].to_numpy())
    return out


# ---------------------------------------------------------------- driver ---------------
def passes(table: dict, margin: float, chance_margin: float) -> dict:
    """Pass/fail rule from the plan: beats shortcut on controls 2 and 3 and the
    background-only model is near chance."""
    c2 = table["resolution_matched"]
    c3a, c3b = table["cross_resolution"]["a_to_b"], table["cross_resolution"]["b_to_a"]
    c4 = table["background_only"]
    short2 = c2["shortcut_size_and_intensity"]["balanced_accuracy"]
    beats2 = c2["model"]["balanced_accuracy"] > short2 + margin
    beats3 = all(
        d["model"]["balanced_accuracy"] > d["shortcut_size_and_intensity"]["balanced_accuracy"] + margin
        for d in (c3a, c3b)
    )
    bg_near_chance = c4["balanced_accuracy"] <= 0.5 + chance_margin
    return {
        "beats_shortcut_resolution_matched": bool(beats2),
        "beats_shortcut_cross_resolution": bool(beats3),
        "background_only_near_chance": bool(bg_near_chance),
        "passes": bool(beats2 and beats3 and bg_near_chance),
    }


def run_controls(geometry: str, model_name: str, params: dict, cfg: dict | None = None,
                 n_jobs: int = 4) -> dict:
    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    pool = load_pool(cfg)
    P = C.load_particles(geometry, cfg, pool)
    feats = feature_columns(ccfg["feature_groups"])
    seed = ccfg["seed"]
    meta = _meta_table(pool, cfg)
    cc = ccfg["controls"]
    a, b = cc["resolution_matched"]
    matched = set(cc["resolution_matched"])

    sub_rel = set(pool.images[res_key(pool.images).isin(matched)]["rel_path"])
    t = {
        "geometry": geometry, "model": model_name, "params": params,
        "shortcut_cv_all_folds": {
            "size_only": shortcut_cv(meta, pool, SIZE_COLS, seed),
            "size_and_intensity": shortcut_cv(meta, pool, SIZE_COLS + STAT_COLS, seed),
        },
        "resolution_matched": {
            "resolutions": sorted(matched),
            "model": resolution_matched(P, pool, ccfg, model_name, params, feats, matched),
            "shortcut_size_and_intensity": shortcut_cv(
                meta, pool, SIZE_COLS + STAT_COLS, seed, subset=sub_rel),
            "shortcut_size_only": shortcut_cv(meta, pool, SIZE_COLS, seed, subset=sub_rel),
        },
        "cross_resolution": {
            k: {
                "train": x, "test": y,
                "model": cross_resolution(P, pool, ccfg, model_name, params, feats, x, y),
                "shortcut_size_and_intensity": shortcut_cross(
                    meta, x, y, SIZE_COLS + STAT_COLS, seed),
            }
            for k, (x, y) in {"a_to_b": (a, b), "b_to_a": (b, a)}.items()
        },
        "background_only": background_only(pool, cfg, geometry, ccfg),
        "feature_group_ablation": feature_group_ablation(P, pool, ccfg),
        "permutation": permutation_test(P, pool, ccfg, feats, n_jobs=n_jobs),
    }
    t["verdict"] = passes(t, cc["shortcut_margin"], cc["chance_margin"])
    return t
