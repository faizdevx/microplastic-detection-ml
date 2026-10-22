"""Drivers that tie features, CV, controls and the final evaluation together."""
from __future__ import annotations

import json

from mp.config import load_config, load_named, resolve
from mp.eval.metrics import run_meta


def results_dir(cfg: dict):
    d = resolve(cfg["paths"]["results"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_cls(cfg: dict | None = None) -> dict:
    """Tuned group-aware CV for every model and geometry; pick the model on CV only."""
    from mp.models import classifier as C

    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    out = {"meta": run_meta(), "geometries": {}}
    for g in ccfg["geometries"]:
        res = C.main_cv(g, cfg)
        for name, r in res.items():
            r["_oof"].to_csv(
                resolve(cfg["paths"]["groups"]).parent / f"oof_{g}_{name}.csv", index=False)
            r.pop("_oof")
        out["geometries"][g] = res
    main = out["geometries"][ccfg["main_geometry"]]
    best = max(main, key=lambda n: main[n]["image_level"]["accuracy"])
    out["selected"] = {
        "geometry": ccfg["main_geometry"], "model": best,
        "params": main[best]["modal_params"],
        "rule": "highest pooled image-level CV accuracy on the main geometry; "
                "hyper-parameters = most frequent inner-CV choice over the outer folds",
    }
    (results_dir(cfg) / "phase2_cv.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    print("selected:", out["selected"]["model"], out["selected"]["params"])
    return out


def controls(cfg: dict | None = None, n_jobs: int = 4) -> dict:
    from mp.eval import controls as K

    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    cv = json.loads((results_dir(cfg) / "phase2_cv.json").read_text(encoding="utf8"))
    sel = cv["selected"]
    out = {"meta": run_meta(), "selected": sel, "geometries": {}}
    for g in ccfg["geometries"]:
        out["geometries"][g] = K.run_controls(g, sel["model"], sel["params"], cfg, n_jobs)
        print(g, out["geometries"][g]["verdict"])
    (results_dir(cfg) / "phase2_controls.json").write_text(
        json.dumps(out, indent=2), encoding="utf8")
    return out
