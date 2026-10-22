"""Final hold-out evaluation (once) and the generated phase-2 summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mp.config import load_config, load_named, resolve  # noqa: E402
from mp.data.dataset import load_pool  # noqa: E402
from mp.eval import metrics as M  # noqa: E402
from mp.imaging import features as F  # noqa: E402
from mp.imaging.features import feature_columns  # noqa: E402
from mp.imaging.segment import segment  # noqa: E402
from mp.models import classifier as C  # noqa: E402


def holdout_path(cfg: dict) -> Path:
    return resolve(cfg["paths"]["results"]) / "phase2_holdout.json"


def guard_holdout(path: Path, i_know: bool) -> None:
    """Refuse to evaluate the hold-out a second time unless explicitly overridden."""
    if path.exists() and not i_know:
        raise SystemExit(
            f"{path} already exists: the hold-out was evaluated once. "
            "Re-evaluating it is selection on the test set; pass --i-know to override."
        )


def _failure_gallery(errors: pd.DataFrame, cfg: dict, ccfg: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = resolve(cfg["paths"]["extract_dir"])
    names = []
    for rank, (_, r) in enumerate(errors.iterrows(), start=1):
        a = F.prepare(root / r["rel_path"], ccfg["main_geometry"], ccfg["preprocess"])
        labels = segment(a["bgsub"], ccfg["segmentation"])
        vis = a["geo"][:, :, ::-1].copy()
        for i in range(1, int(labels.max()) + 1):
            cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
        true = M.CLASS_NAMES[int(r["y"])]
        pred = M.CLASS_NAMES[int(r["prob"] >= 0.5)]
        cv2.putText(vis, f"true {true} -> pred {pred} (p_nylon={r['prob']:.2f})", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        name = f"{rank:02d}_{Path(r['rel_path']).stem}.png"
        cv2.imwrite(str(out_dir / name), vis)
        names.append(name)
    return names


def run_holdout(i_know: bool = False, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    res_dir = resolve(cfg["paths"]["results"])
    path = holdout_path(cfg)
    guard_holdout(path, i_know)

    cv = json.loads((res_dir / "phase2_cv.json").read_text(encoding="utf8"))
    sel = cv["selected"]
    pool = load_pool(cfg)
    P = C.load_particles(sel["geometry"], cfg, pool)
    feats = feature_columns(ccfg["feature_groups"])
    img = C.eval_split(sel["model"], sel["params"], P, pool.cv(), pool.holdout(), feats,
                       ccfg["seed"])
    ok = img.dropna(subset=["prob"]).rename(columns={})
    y, prob = ok["y"].to_numpy(), ok["prob"].to_numpy()
    point = M.point_metrics(y, prob)
    ci = M.group_bootstrap_ci(ok, n=ccfg["holdout"]["bootstrap"], seed=ccfg["seed"])

    figs = res_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rel = M.reliability(y, prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(rel["mean_prob"], rel["frac_pos"], "o-")
    ax.set_xlabel("predicted P(NYLON)")
    ax.set_ylabel("observed NYLON fraction")
    ax.set_title(f"Hold-out reliability (ECE {point['ece']:.3f})")
    fig.tight_layout()
    fig.savefig(figs / "phase2_reliability.png", dpi=120)
    plt.close(fig)

    conf_err = ok[(ok["prob"] >= 0.5).astype(int) != ok["y"]].copy()
    conf_err["conf"] = np.maximum(conf_err["prob"], 1 - conf_err["prob"])
    worst = conf_err.sort_values("conf", ascending=False).head(ccfg["holdout"]["failure_gallery"])
    gallery = _failure_gallery(worst, cfg, ccfg, figs / "phase2_failures")

    mc = None
    yolo_preds = resolve(cfg["paths"]["groups"]).parent.parent / "yolo" / "preds_holdout.csv"
    if yolo_preds.exists():
        yp = pd.read_csv(yolo_preds).dropna(subset=["prob"]).merge(
            ok[["rel_path", "prob"]], on="rel_path", suffixes=("_yolo", "_a"))
        mc = M.mcnemar(yp["y"].to_numpy(), (yp["prob_a"] >= 0.5).astype(int).to_numpy(),
                       (yp["prob_yolo"] >= 0.5).astype(int).to_numpy())

    out = {
        "meta": M.run_meta(),
        "model": sel, "n_holdout_images": int(len(img)),
        "n_without_particles": int(img["prob"].isna().sum()),
        "metrics": point, "ci95_group_bootstrap": ci,
        "n_bootstrap": ccfg["holdout"]["bootstrap"],
        "failure_gallery": gallery,
        "mcnemar_vs_yolo": mc if mc else "YOLO hold-out predictions not available",
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf8")
    return out


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def write_summary(cfg: dict | None = None) -> Path:
    """results/phase2_summary.md, generated from the JSON files only."""
    cfg = cfg or load_config()
    d = resolve(cfg["paths"]["results"])
    cv = json.loads((d / "phase2_cv.json").read_text(encoding="utf8"))
    ctl_p, ho_p = d / "phase2_controls.json", d / "phase2_holdout.json"
    ctl = json.loads(ctl_p.read_text(encoding="utf8")) if ctl_p.exists() else None
    ho = json.loads(ho_p.read_text(encoding="utf8")) if ho_p.exists() else None
    sel = cv["selected"]
    L = ["## Polymer identification: results (generated)", ""]
    L += [f"Selected on CV only: **{sel['model']}** `{sel['params']}` on geometry "
          f"`{sel['geometry']}`.", ""]
    L += ["### Cross-validation (5 group-aware folds, image level)", "",
          "| Geometry | Model | Pooled accuracy | Fold mean ± std | Macro-F1 | ROC-AUC |",
          "|---|---|---|---|---|---|"]
    for g, models in cv["geometries"].items():
        for n, v in models.items():
            il = v["image_level"]
            L.append(f"| {g} | {n} | {il['accuracy']:.3f} | {v['fold_mean']:.3f} ± "
                     f"{v['fold_std']:.3f} | {il['macro_f1']:.3f} | {_fmt(il['roc_auc'])} |")
    if ctl:
        L += ["", "### Controls (balanced accuracy unless noted)", "",
              "| Geometry | Shortcut CV (size+intensity) | Res.-matched model / shortcut | "
              "640→1280 model / shortcut | 1280→640 model / shortcut | Background-only | "
              "Permutation p | Passes |", "|---|---|---|---|---|---|---|---|"]
        for g, t in ctl["geometries"].items():
            ca, cb = t["cross_resolution"]["a_to_b"], t["cross_resolution"]["b_to_a"]
            sc = t["shortcut_cv_all_folds"]["size_and_intensity"]["balanced_accuracy"]
            rm = t["resolution_matched"]
            L.append(
                f"| {g} | {sc:.3f} | {rm['model']['balanced_accuracy']:.3f} / "
                f"{rm['shortcut_size_and_intensity']['balanced_accuracy']:.3f} | "
                f"{ca['model']['balanced_accuracy']:.3f} / "
                f"{ca['shortcut_size_and_intensity']['balanced_accuracy']:.3f} | "
                f"{cb['model']['balanced_accuracy']:.3f} / "
                f"{cb['shortcut_size_and_intensity']['balanced_accuracy']:.3f} | "
                f"{t['background_only']['balanced_accuracy']:.3f} | "
                f"{t['permutation']['p_value']:.3f} | {'yes' if t['verdict']['passes'] else 'no'} |")
        L += ["", "Feature-group ablation (HistGB, accuracy):", "",
              "| Geometry | " + " | ".join(next(iter(ctl["geometries"].values()))
                                            ["feature_group_ablation"]) + " |",
              "|---|" + "---|" * len(next(iter(ctl["geometries"].values()))["feature_group_ablation"])]
        for g, t in ctl["geometries"].items():
            L.append(f"| {g} | " + " | ".join(
                f"{v['accuracy']:.3f}" for v in t["feature_group_ablation"].values()) + " |")
    if ho:
        m, ci = ho["metrics"], ho["ci95_group_bootstrap"]
        L += ["", "### Hold-out (evaluated once)", "",
              "| Metric | Value | 95 % CI (group bootstrap) |", "|---|---|---|"]
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"):
            c = ci.get(k)
            L.append(f"| {k} | {_fmt(m[k])} | "
                     f"{'n/a' if not c else f'{c[0]:.3f} - {c[1]:.3f}'} |")
        L += ["", f"Confusion matrix (rows true PVC/NYLON): {m['confusion_matrix']}; "
                  f"ECE {m['ece']:.3f}; {ho['n_holdout_images']} images."]
    else:
        L += ["", "_Hold-out not evaluated yet._"]
    out = d / "phase2_summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf8")
    return out


def main(argv=None) -> int:
    try:
        run_holdout(i_know="--i-know" in (argv or sys.argv))
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1
    return 0
