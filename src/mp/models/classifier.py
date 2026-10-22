"""Baseline A: per-particle classifier aggregated to images, with group-aware CV."""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mp.config import load_config, load_named
from mp.data.dataset import Pool, load_pool
from mp.eval.metrics import point_metrics
from mp.imaging.features import feature_columns, particles_path

MODEL_NAMES = ("logreg", "random_forest", "hist_gb")
IMG_COLS = ["rel_path", "y", "group_id", "weight", "fold", "width", "height", "class"]


def load_particles(geometry: str, cfg: dict | None = None, pool: Pool | None = None) -> pd.DataFrame:
    """Particle table joined with image labels/groups. Only pool images are kept."""
    cfg = cfg or load_config()
    pool = pool or load_pool(cfg)
    raw = pd.read_csv(particles_path(cfg, geometry))
    return raw.merge(pool.images[IMG_COLS], on="rel_path", how="inner")


def make_model(name: str, params: dict, seed: int):
    p = dict(params)
    if name == "logreg":
        clf = LogisticRegression(max_iter=3000, random_state=seed, **p)
        steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                 ("clf", clf)]
    elif name == "random_forest":
        p.setdefault("n_estimators", 200)
        p.setdefault("n_jobs", -1)
        clf = RandomForestClassifier(random_state=seed, **p)
        steps = [("imp", SimpleImputer(strategy="median")), ("clf", clf)]
    elif name == "hist_gb":
        p.setdefault("max_iter", 100)
        clf = HistGradientBoostingClassifier(random_state=seed, **p)
        steps = [("clf", clf)]  # handles NaN natively
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline(steps)


def particle_weights(P: pd.DataFrame) -> np.ndarray:
    """Image weight (1/group size, class-balanced) spread over its particles, mean 1."""
    n = P.groupby("rel_path")["rel_path"].transform("size").to_numpy()
    w = P["weight"].to_numpy() / n
    return w / w.mean()


def fit_proba(name, params, Ptr, Pva, feats, seed) -> np.ndarray:
    model = make_model(name, params, seed)
    model.fit(Ptr[feats], Ptr["y"], clf__sample_weight=particle_weights(Ptr))
    return model.predict_proba(Pva[feats])[:, 1]


def aggregate(Pva: pd.DataFrame, proba: np.ndarray, images: pd.DataFrame) -> pd.DataFrame:
    """Mean particle probability per image. Images without particles are returned with
    prob NaN so coverage can be reported."""
    agg = (
        pd.DataFrame({"rel_path": Pva["rel_path"].to_numpy(), "prob": proba})
        .groupby("rel_path", as_index=False)["prob"].mean()
    )
    out = images[["rel_path", "y", "group_id", "fold", "class", "width", "height"]].merge(
        agg, on="rel_path", how="left"
    )
    out["n_particles"] = out["rel_path"].map(Pva.groupby("rel_path").size()).fillna(0).astype(int)
    return out


def _grid(spec: dict) -> list[dict]:
    grid = spec.get("grid", {})
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))] or [{}]
    return [{**spec.get("fixed", {}), **c} for c in combos]


def tune(name, spec, Ptr, images_tr, feats, inner_folds, seed) -> dict:
    """Pick hyper-parameters by inner GroupKFold on image-level accuracy."""
    combos = _grid(spec)
    if len(combos) == 1:
        return combos[0]
    gkf = GroupKFold(n_splits=inner_folds)
    scores = np.zeros(len(combos))
    for tr_idx, va_idx in gkf.split(images_tr, groups=images_tr["group_id"]):
        tr_imgs, va_imgs = images_tr.iloc[tr_idx], images_tr.iloc[va_idx]
        a = Ptr[Ptr["rel_path"].isin(tr_imgs["rel_path"])]
        b = Ptr[Ptr["rel_path"].isin(va_imgs["rel_path"])]
        for i, params in enumerate(combos):
            img = aggregate(b, fit_proba(name, params, a, b, feats, seed), va_imgs).dropna(subset=["prob"])
            scores[i] += ((img["prob"] >= 0.5).astype(int) == img["y"]).mean()
    return combos[int(np.argmax(scores))]


def eval_split(
    name: str, params: dict, P: pd.DataFrame, train_imgs: pd.DataFrame,
    val_imgs: pd.DataFrame, feats: list[str], seed: int,
) -> pd.DataFrame:
    """Fit on train images, predict val images (image-level table, prob NaN if no particles)."""
    Ptr = P[P["rel_path"].isin(train_imgs["rel_path"])]
    Pva = P[P["rel_path"].isin(val_imgs["rel_path"])]
    if Ptr["y"].nunique() < 2 or len(Pva) == 0:
        out = val_imgs[["rel_path", "y", "group_id", "fold", "class", "width", "height"]].copy()
        out["prob"], out["n_particles"] = np.nan, 0
        return out
    return aggregate(Pva, fit_proba(name, params, Ptr, Pva, feats, seed), val_imgs)


def run_cv(
    name: str, P: pd.DataFrame, pool: Pool, ccfg: dict, feats: list[str],
    do_tune: bool = True, params: dict | None = None, subset: pd.Index | None = None,
) -> dict:
    """Outer CV over the splits.json folds; returns OOF image table + per-fold records."""
    seed = ccfg["seed"]
    spec = ccfg["models"][name]
    oof, fold_rows, best = [], [], []
    for k in range(pool.n_folds):
        tr, va = pool.fold_split(k)
        if subset is not None:
            tr, va = tr[tr["rel_path"].isin(subset)], va[va["rel_path"].isin(subset)]
        if do_tune and params is None:
            chosen = tune(name, spec, P[P["rel_path"].isin(tr["rel_path"])], tr, feats,
                          ccfg["inner_folds"], seed)
        else:
            chosen = params if params is not None else _grid(spec)[0]
        best.append(chosen)
        img = eval_split(name, chosen, P, tr, va, feats, seed)
        img["outer_fold"] = k
        oof.append(img)
        ok = img.dropna(subset=["prob"])
        fold_rows.append({
            "fold": k, "params": chosen, "n_val_images": int(len(va)),
            "n_val_no_particles": int(img["prob"].isna().sum()),
            "image_accuracy": float(((ok["prob"] >= 0.5).astype(int) == ok["y"]).mean()),
        })
    oof = pd.concat(oof, ignore_index=True)
    return {"oof": oof, "folds": fold_rows, "best_params": best}


def summarize_oof(oof: pd.DataFrame) -> dict:
    ok = oof.dropna(subset=["prob"])
    out = point_metrics(ok["y"].to_numpy(), ok["prob"].to_numpy())
    out["n_images_total"] = int(len(oof))
    out["n_images_without_particles"] = int(oof["prob"].isna().sum())
    return out


def modal_params(best: list[dict]) -> dict:
    keyed = Counter(repr(sorted(b.items())) for b in best)
    top = keyed.most_common(1)[0][0]
    return next(b for b in best if repr(sorted(b.items())) == top)


def main_cv(geometry: str | None = None, cfg: dict | None = None) -> dict:
    """Full tuned CV for all models on one geometry."""
    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    geometry = geometry or ccfg["main_geometry"]
    pool = load_pool(cfg)
    P = load_particles(geometry, cfg, pool)
    feats = feature_columns(ccfg["feature_groups"])
    res = {}
    for name in MODEL_NAMES:
        r = run_cv(name, P, pool, ccfg, feats)
        accs = [f["image_accuracy"] for f in r["folds"]]
        res[name] = {
            "image_level": summarize_oof(r["oof"]),
            "fold_image_accuracy": accs,
            "fold_mean": float(np.mean(accs)),
            "fold_std": float(np.std(accs)),
            "folds": r["folds"],
            "modal_params": modal_params(r["best_params"]),
            "_oof": r["oof"],
        }
    return res
