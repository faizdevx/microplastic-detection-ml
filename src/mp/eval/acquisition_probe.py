"""Can acquisition cues (exposure, resolution) be read from a variant's features?

Two checks per variant, both group-aware and on CV images only:
  probe    image-level mean features -> predict the exposure tertile and the resolution,
           overall and within each class (so the class-brightness link is removed)
  coupling Spearman correlation between the out-of-fold P(NYLON) and image brightness,
           within each class
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from mp.eval.controls import res_key


def _cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
    if len(np.unique(y)) < 2 or len(y) < 20:
        return float("nan")
    n = min(5, len(np.unique(groups)))
    clf = HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits=n))
    return float((pred == y).mean())


def _chance(y: np.ndarray) -> float:
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / c.sum())


def image_matrix(variant_name: str, feats: list | None, data, emb: dict | None) -> pd.DataFrame:
    """One row per image: mean of the variant's particle features (or CNN embedding)."""
    imgs = data.images.set_index("rel_path")
    if emb is not None:
        m = pd.DataFrame.from_dict(emb, orient="index")
    else:
        m = data.real.groupby("rel_path")[feats].mean()
    return m.join(imgs[["y", "group_id", "class"]], how="inner").dropna()


def probe_variant(name: str, mat: pd.DataFrame, data, rcfg: dict) -> dict:
    seed = rcfg["seed"]
    bins = rcfg["probe"]["exposure_bins"]
    cols = [c for c in mat.columns if c not in ("y", "group_id", "class")]
    bright = data.bright.reindex(mat.index)
    res = res_key(data.images.set_index("rel_path")).reindex(mat.index)
    out = {"n_images": int(len(mat))}

    def target_exposure(idx):
        return pd.qcut(bright.loc[idx], bins, labels=False, duplicates="drop").to_numpy()

    X = mat[cols].to_numpy(float)
    y_exp = target_exposure(mat.index)
    out["exposure_overall"] = {"accuracy": _cv_accuracy(X, y_exp, mat["group_id"].to_numpy(), seed),
                               "chance": _chance(y_exp)}
    within = []
    for cls, sub in mat.groupby("class"):
        ye = target_exposure(sub.index)
        within.append((_cv_accuracy(sub[cols].to_numpy(float), ye, sub["group_id"].to_numpy(), seed),
                       _chance(ye)))
    out["exposure_within_class"] = {"accuracy": float(np.nanmean([a for a, _ in within])),
                                    "chance": float(np.mean([c for _, c in within]))}
    counts = res.value_counts()
    keep = res.isin(counts[counts >= rcfg["probe"]["min_images_per_resolution"]].index)
    yr = res[keep].to_numpy()
    out["resolution"] = {"accuracy": _cv_accuracy(mat.loc[keep, cols].to_numpy(float), yr,
                                                  mat.loc[keep, "group_id"].to_numpy(), seed),
                         "chance": _chance(yr), "n_resolutions": int(len(np.unique(yr)))}
    return out


def prediction_coupling(oof: pd.DataFrame, data) -> dict:
    """Within-class Spearman correlation of P(NYLON) with brightness (0 = no coupling)."""
    ok = oof.dropna(subset=["prob"]).copy()
    ok["bright"] = ok["rel_path"].map(data.bright)
    out = {}
    for cls, sub in ok.groupby("class"):
        rho = spearmanr(sub["prob"], sub["bright"]).statistic
        out[cls] = float(rho)
    out["mean_abs"] = float(np.mean([abs(v) for v in out.values()]))
    return out


def run(res: dict, feats_by_variant: dict, data, rcfg: dict, embeddings: dict) -> dict:
    out = {}
    for name in res["oof"]:
        emb = embeddings.get(name)
        mat = image_matrix(name, feats_by_variant.get(name), data, emb)
        out[name] = {"probe": probe_variant(name, mat, data, rcfg),
                     "prediction_coupling": prediction_coupling(res["oof"][name], data)}
    out["note"] = ("V2 changes training, not features, so its feature probe equals the baseline's; "
                   "its coupling row shows the effect of exposure randomisation on predictions")
    return out
