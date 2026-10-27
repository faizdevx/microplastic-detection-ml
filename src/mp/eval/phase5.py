"""Phase 5: baseline, V1, V2 and V3 evaluated with the same controls (CV pool only)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from mp.config import load_config, repo_root, resolve
from mp.data.dataset import load_pool
from mp.eval.controls import acc_bal, res_key
from mp.eval.metrics import run_meta
from mp.imaging import invariant as I
from mp.imaging.features import FEATURE_GROUPS
from mp.models import classifier as C
from mp.models import patch_cnn

IMG_COLS = ["rel_path", "y", "group_id", "fold", "class", "width", "height"]


def load_robust_config() -> dict:
    return yaml.safe_load(open(repo_root() / "configs" / "robustness.yaml", encoding="utf8"))


@dataclass
class Data:
    images: pd.DataFrame
    n_folds: int
    real: pd.DataFrame
    pseudo: pd.DataFrame
    aug: pd.DataFrame
    patches: dict = field(default_factory=dict)       # kind -> (array, idx dataframe)
    bright: pd.Series | None = None

    def fold_split(self, k: int, subset=None):
        tr, va = self.images[self.images["fold"] != k], self.images[self.images["fold"] == k]
        if subset is not None:
            tr, va = tr[tr["rel_path"].isin(subset)], va[va["rel_path"].isin(subset)]
        return tr, va


def load_data(cfg: dict | None = None, with_patches: bool = True) -> Data:
    cfg = cfg or load_config()
    pool = load_pool(cfg)
    imgs = pool.cv()[IMG_COLS].reset_index(drop=True)      # hold-out rows are never loaded
    d = I.data_dir()
    keep = set(imgs["rel_path"])
    real = pd.read_csv(d / "real.csv.gz")
    pseudo = pd.read_csv(d / "pseudo.csv.gz")
    aug = pd.read_csv(d / "aug.csv.gz")
    for t in (real, pseudo, aug):
        assert set(t["rel_path"]) <= keep, "table contains images outside the CV pool"
    patches = {}
    if with_patches:
        for kind in ("real", "pseudo"):
            patches[kind] = (np.load(d / f"patches_{kind}.npy"),
                             pd.read_csv(d / f"patches_{kind}_idx.csv"))
    stats = pd.read_csv(resolve(cfg["paths"]["image_stats"]))
    stats = stats[stats["rel_path"].isin(keep)].set_index("rel_path")
    bright = stats[["mean_R", "mean_G", "mean_B"]].mean(axis=1)
    return Data(imgs, pool.n_folds, real, pseudo, aug, patches, bright)


def image_weights(tr: pd.DataFrame, ymap: dict | None = None) -> pd.Series:
    """1 / group size within the training images; equal total weight per class; mean 1."""
    y = tr["rel_path"].map(ymap) if ymap is not None else tr["y"]
    w = 1.0 / tr.groupby("group_id")["group_id"].transform("size")
    w = w / w.groupby(y.to_numpy()).transform("sum")
    w = w / w.mean()
    return pd.Series(w.to_numpy(), index=tr["rel_path"].to_numpy())


def aggregate(rows: pd.DataFrame, prob: np.ndarray, va: pd.DataFrame) -> pd.DataFrame:
    agg = (pd.DataFrame({"rel_path": rows["rel_path"].to_numpy(), "prob": prob})
           .groupby("rel_path", as_index=False)["prob"].mean())
    out = va[IMG_COLS].merge(agg, on="rel_path", how="left")
    out["n_particles"] = out["rel_path"].map(rows.groupby("rel_path").size()).fillna(0).astype(int)
    return out


# ------------------------------------------------------------------ variants -----------
@dataclass
class TabularVariant:
    name: str
    feats: list
    use_aug: bool
    params: dict
    seed: int
    iters: int | None = None

    def predict(self, data: Data, tr, va, table="real", ymap=None, feats=None, iters=None):
        feats = feats or self.feats
        T = data.real if table == "real" else data.pseudo
        rows = T[T["rel_path"].isin(tr["rel_path"])]
        if self.use_aug and table == "real":
            a = data.aug
            rows = pd.concat([rows, a[a["rel_path"].isin(tr["rel_path"])]], ignore_index=True)
        w_img = image_weights(tr, ymap)
        y_img = tr.set_index("rel_path")["y"] if ymap is None else pd.Series(ymap)
        yy = rows["rel_path"].map(y_img).to_numpy()
        n = rows.groupby("rel_path")["rel_path"].transform("size").to_numpy()
        w = rows["rel_path"].map(w_img).to_numpy() / n
        params = dict(self.params)
        if iters or self.iters:
            params["max_iter"] = iters or self.iters
        if len(np.unique(yy)) < 2 or not len(rows):
            return aggregate(rows.iloc[:0], np.array([]), va)
        model = C.make_model("hist_gb", params, self.seed)
        model.fit(rows[feats], yy, clf__sample_weight=w / w.mean())
        vrows = T[T["rel_path"].isin(va["rel_path"])]
        prob = model.predict_proba(vrows[feats])[:, 1] if len(vrows) else np.array([])
        return aggregate(vrows, prob, va)


@dataclass
class CnnVariant:
    name: str
    epochs: int
    seed: int
    emb: dict = field(default_factory=dict)
    use_aug: bool = False

    def predict(self, data: Data, tr, va, table="real", ymap=None, epochs=None, **_):
        X, idx = data.patches[table]
        tr_m = idx["rel_path"].isin(tr["rel_path"]).to_numpy()
        va_m = idx["rel_path"].isin(va["rel_path"]).to_numpy()
        w_img = image_weights(tr, ymap)
        y_img = tr.set_index("rel_path")["y"] if ymap is None else pd.Series(ymap)
        r = idx.loc[tr_m, "rel_path"]
        yy = r.map(y_img).to_numpy()
        n = r.map(r.value_counts()).to_numpy()
        w = r.map(w_img).to_numpy() / n
        if len(np.unique(yy)) < 2:
            return aggregate(idx.iloc[:0], np.array([]), va)
        net = patch_cnn.fit(X[tr_m], yy, w, epochs or self.epochs, self.seed)
        vrows = idx[va_m]
        prob, emb = patch_cnn.predict(net, X[va_m]) if va_m.any() else (np.array([]), None)
        out = aggregate(vrows, prob, va)
        if emb is not None and table == "real":
            e = pd.DataFrame(emb)
            e["rel_path"] = vrows["rel_path"].to_numpy()
            self.emb.update(e.groupby("rel_path").mean().T.to_dict("list"))
        return out


# ---------------------------------------------------------------------- controls -------
def run_cv(variant, data: Data, table="real", ymap=None, subset=None, **kw) -> pd.DataFrame:
    parts = []
    for k in range(data.n_folds):
        tr, va = data.fold_split(k, subset)
        o = variant.predict(data, tr, va, table=table, ymap=ymap, **kw)
        o["outer_fold"] = k
        parts.append(o)
    return pd.concat(parts, ignore_index=True)


def _acc(oof: pd.DataFrame) -> dict:
    ok = oof.dropna(subset=["prob"])
    return acc_bal(ok["y"].to_numpy(), ok["prob"].to_numpy())


def cross_res(variant, data: Data, a: str, b: str) -> dict:
    te = data.images[res_key(data.images) == b]
    tr = data.images[(res_key(data.images) == a) & ~data.images["group_id"].isin(te["group_id"])]
    return _acc(variant.predict(data, tr, te))


def permuted_labels(images: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    g = images.drop_duplicates("group_id")[["group_id", "y"]].reset_index(drop=True)
    m = dict(zip(g["group_id"], rng.permutation(g["y"].to_numpy())))
    return dict(zip(images["rel_path"], images["group_id"].map(m)))


def _perm_tab(variant, data, seed, iters):
    return _acc(run_cv(variant, data, ymap=permuted_labels(data.images, seed), iters=iters))["accuracy"]


def permutation_tabular(variant, data: Data, n: int, iters: int, obs: float) -> dict:
    null = np.array(Parallel(n_jobs=4)(delayed(_perm_tab)(variant, data, 1000 + i, iters)
                                        for i in range(n)))
    return {"observed_accuracy": obs, "null_mean": float(null.mean()), "null_max": float(null.max()),
            "n_permutations": n, "p_value": float((1 + (null >= obs).sum()) / (1 + n))}


def permutation_cnn(variant, data: Data, n: int, epochs: int, seed: int) -> dict:
    gids = data.images["group_id"].unique()
    obs, null = [], []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        test_g = set(rng.choice(gids, size=len(gids) // 5, replace=False))
        te = data.images[data.images["group_id"].isin(test_g)]
        tr = data.images[~data.images["group_id"].isin(test_g)]
        obs.append(_acc(variant.predict(data, tr, te, epochs=epochs))["accuracy"])
        ym = permuted_labels(data.images, seed + 500 + i)
        null.append(_acc(variant.predict(data, tr, te, ymap=ym, epochs=epochs))["accuracy"])
    o, null = float(np.mean(obs)), np.array(null)
    return {"observed_accuracy": o, "null_mean": float(null.mean()), "null_max": float(null.max()),
            "n_permutations": n, "p_value": float((1 + (null >= o).sum()) / (1 + n)),
            "protocol": "one 80/20 group split per permutation, 1 epoch"}


def ablation(variant: TabularVariant, data: Data, subsets: dict) -> dict:
    return {name: _acc(run_cv(variant, data, feats=cols)) for name, cols in subsets.items()}


def evaluate_variant(variant, data: Data, rcfg: dict, subsets: dict | None = None) -> dict:
    cc = rcfg["controls"]
    a, b = cc["resolution_matched"]
    sub = set(data.images[res_key(data.images).isin({a, b})]["rel_path"])
    oof = run_cv(variant, data)
    ok = oof.dropna(subset=["prob"])
    fold_acc = [float(((g["prob"] >= 0.5).astype(int) == g["y"]).mean())
                for _, g in ok.groupby("outer_fold")]
    out = {"cv": {**_acc(oof), "fold_accuracy_mean": float(np.mean(fold_acc)),
                  "fold_accuracy_std": float(np.std(fold_acc)),
                  "n_without_particles": int(oof["prob"].isna().sum())}}
    out["resolution_matched"] = _acc(run_cv(variant, data, subset=sub))
    out["cross_resolution"] = {"a_to_b": {"train": a, "test": b, **cross_res(variant, data, a, b)},
                               "b_to_a": {"train": b, "test": a, **cross_res(variant, data, b, a)}}
    out["pseudo_particles"] = _acc(run_cv(variant, data, table="pseudo"))
    if isinstance(variant, TabularVariant):
        out["permutation"] = permutation_tabular(variant, data, cc["permutations_tabular"],
                                                 cc["permutation_model_iter"], out["cv"]["accuracy"])
        if subsets:
            out["feature_ablation"] = ablation(variant, data, subsets)
    else:
        out["permutation"] = permutation_cnn(variant, data, cc["permutations_cnn"],
                                             rcfg["cnn"]["permutation_epochs"], rcfg["seed"])
    return out, oof


def make_variants(rcfg: dict) -> dict:
    tab = rcfg["tabular_model"]["params"]
    seed = rcfg["seed"]
    return {
        "baseline": TabularVariant("baseline", I.BASE_FEATURES, False, tab, seed),
        "V1_local_normalisation": TabularVariant("V1", I.V1_FEATURES, False, tab, seed),
        "V2_exposure_randomisation": TabularVariant("V2", I.BASE_FEATURES, True, tab, seed),
        "V3_patch_cnn": CnnVariant("V3", rcfg["cnn"]["epochs"], seed),
    }


def feature_subsets() -> dict:
    g = FEATURE_GROUPS
    return {"baseline": {"shape": g["shape"], "ratio": g["ratio"], "intensity_norm": g["intensity_norm"],
                         "intensity_raw": g["intensity_raw"]},
            "V1_local_normalisation": {"shape": g["shape"], "ratio": g["ratio"], "v1_local": I.V1_EXTRA},
            "V2_exposure_randomisation": {"shape": g["shape"], "ratio": g["ratio"],
                                          "intensity_norm": g["intensity_norm"],
                                          "intensity_raw": g["intensity_raw"]}}


def run_all(only: list | None = None) -> dict:
    rcfg = load_robust_config()
    data = load_data()
    variants = make_variants(rcfg)
    subs = feature_subsets()
    res, oofs, embs = {}, {}, {}
    for name, v in variants.items():
        if only and name not in only:
            continue
        print("evaluating", name, flush=True)
        r, oof = evaluate_variant(v, data, rcfg, subs.get(name))
        res[name], oofs[name] = r, oof
        if isinstance(v, CnnVariant):
            embs[name] = v.emb
    out_dir = resolve("results")
    (out_dir / "phase5_variants.json").write_text(
        json.dumps({"meta": run_meta(), "config": rcfg, "n_cv_images": int(len(data.images)),
                    "results": res}, indent=2), encoding="utf8")
    for n, o in oofs.items():
        o.to_csv(I.data_dir() / f"oof_{n}.csv", index=False)
    return {"results": res, "oof": oofs, "emb": embs, "data": data, "rcfg": rcfg}


def feature_set_controls(data: Data, rcfg: dict) -> dict:
    """Which kinds of feature carry polymer information beyond the acquisition cues?

    Each feature set gets the cross-validation, cross-resolution and background-disc controls.
    Shape features computed on the discs are class-independent by construction, so their
    background-disc score is a check on the control itself (it should sit at chance).
    """
    cc = rcfg["controls"]
    a, b = cc["resolution_matched"]
    sets = {
        "shape": FEATURE_GROUPS["shape"],
        "colour_ratios": FEATURE_GROUPS["ratio"],
        "shape+colour_ratios": FEATURE_GROUPS["shape"] + FEATURE_GROUPS["ratio"],
        "normalised_intensity": FEATURE_GROUPS["intensity_norm"],
        "raw_intensity": FEATURE_GROUPS["intensity_raw"],
        "v1_local": I.V1_EXTRA,
    }
    base = TabularVariant("subset", [], False, rcfg["tabular_model"]["params"], rcfg["seed"])
    out = {}
    for name, cols in sets.items():
        v = TabularVariant(name, list(cols), False, base.params, base.seed)
        oof = run_cv(v, data)
        out[name] = {
            "n_features": len(cols),
            "cv": _acc(oof),
            "cross_resolution": {"a_to_b": cross_res(v, data, a, b), "b_to_a": cross_res(v, data, b, a)},
            "background_discs": _acc(run_cv(v, data, table="pseudo")),
        }
    return out
