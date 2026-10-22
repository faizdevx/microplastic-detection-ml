"""Permutation importance and SHAP for the selected baseline-A model (CV data only)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from mp.config import load_config, load_named, resolve  # noqa: E402
from mp.data.dataset import load_pool  # noqa: E402
from mp.imaging.features import FEATURE_GROUPS, feature_columns  # noqa: E402
from mp.models import classifier as C  # noqa: E402

MEANING = {
    "area_px": "particle size (px); depends on pixel pitch, not only on polymer",
    "eq_diameter_px": "particle size (px)",
    "circularity": "roundness; fibres and fragments score low",
    "solidity": "how filled-in the outline is (clumps/irregular edges lower it)",
    "convexity": "smoothness of the outline",
    "aspect_ratio": "elongation",
    "ratio_R_over_G": "emission colour: Nile Red shifts with polymer polarity",
    "frac_R": "share of red in the particle (emission colour)",
    "frac_G": "share of green in the particle (emission colour)",
    "contrast_R": "particle brightness above its local surround (exposure dependent)",
    "contrast_ratio_R": "particle/surround brightness ratio",
}


def _meaning(f: str) -> str:
    if f in MEANING:
        return MEANING[f]
    if f.startswith("norm_"):
        return "per-image-normalised channel intensity inside the particle"
    if f.startswith("raw_"):
        return "background-corrected channel intensity (depends on exposure/brightness)"
    if f.startswith("hu"):
        return "shape moment invariant"
    return ""


def explain(cfg: dict | None = None, geometry: str | None = None) -> dict:
    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    res_dir = resolve(cfg["paths"]["results"])
    cv = json.loads((res_dir / "phase2_cv.json").read_text(encoding="utf8"))
    sel = cv["selected"]
    geometry = geometry or sel["geometry"]
    pool = load_pool(cfg)
    P = C.load_particles(geometry, cfg, pool)
    feats = feature_columns(ccfg["feature_groups"])
    tr, va = pool.fold_split(0)  # explanation uses CV data only, never the hold-out
    Ptr = P[P["rel_path"].isin(tr["rel_path"])]
    Pva = P[P["rel_path"].isin(va["rel_path"])]
    model = C.make_model(sel["model"], sel["params"], ccfg["seed"])
    model.fit(Ptr[feats], Ptr["y"], clf__sample_weight=C.particle_weights(Ptr))

    sub = Pva.sample(min(4000, len(Pva)), random_state=ccfg["seed"])
    pi = permutation_importance(model, sub[feats], sub["y"], n_repeats=5,
                                random_state=ccfg["seed"], n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    imp["group"] = imp["feature"].map(group_of)
    imp["meaning"] = imp["feature"].map(_meaning)

    shap_note, shap_top = "not computed", None
    try:
        import shap

        sample = sub[feats].head(1000)
        inner = model.named_steps["clf"]
        X = model[:-1].transform(sample) if len(model.steps) > 1 else sample
        sv = shap.TreeExplainer(inner).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        shap_top = (pd.Series(np.abs(sv).mean(0), index=feats)
                    .sort_values(ascending=False).head(12).round(5).to_dict())
        shap_note = "TreeExplainer on 1000 held-out particles"
    except Exception as exc:  # unsupported model type etc.
        shap_note = f"SHAP unavailable for {sel['model']}: {type(exc).__name__}"

    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"shape": "#4c72b0", "ratio": "#dd8452", "intensity_norm": "#55a868",
              "intensity_raw": "#c44e52"}
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            xerr=top["std"][::-1], color=[colors[g] for g in top["group"][::-1]])
    ax.set_xlabel("permutation importance (accuracy drop, held-out fold 0)")
    ax.set_title(f"{sel['model']} / {geometry}: top features")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    (res_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(res_dir / "figures" / "feature_importance.png", dpi=120)
    plt.close(fig)

    d = {}
    for f in top["feature"]:
        a, b = sub.loc[sub["y"] == 1, f].dropna(), sub.loc[sub["y"] == 0, f].dropna()
        pooled = np.sqrt((a.var() + b.var()) / 2) or np.nan
        d[f] = float((a.mean() - b.mean()) / pooled)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(list(d)[::-1], list(d.values())[::-1], color="#8172b2")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("standardised mean difference (NYLON - PVC), Cohen's d")
    fig.tight_layout()
    fig.savefig(res_dir / "figures" / "feature_effects.png", dpi=120)
    plt.close(fig)

    out = {
        "model": sel["model"], "geometry": geometry, "fit_on": "CV folds 1-4",
        "evaluated_on": "CV fold 0 particles (subsample)",
        "permutation_importance": imp.head(20).round(5).to_dict(orient="records"),
        "importance_by_group": imp.groupby("group")["importance"].sum().round(5).to_dict(),
        "cohens_d_top_features": d, "shap_note": shap_note, "shap_mean_abs_top": shap_top,
    }
    (res_dir / "phase2_explain.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    return out
