"""Summary of the YOLOv11-seg runs: results/yolo_summary.json and results/yolo_summary.md."""
from __future__ import annotations

import json

import numpy as np

from mp.config import resolve

RES = resolve("results")


def _load(key: str) -> dict | None:
    p = RES / f"phase2_yolo_{key}.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def _ms(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}


def _group(keys: list[str]) -> dict | None:
    runs = [r for r in (_load(k) for k in keys) if r]
    if not runs:
        return None
    out = {"n_runs": len(runs),
           "epochs_run": [r["epochs_run"] for r in runs],
           "mask_map50": _ms([r["seg"]["map50"] for r in runs]),
           "mask_map50_95": _ms([r["seg"]["map50_95"] for r in runs]),
           "box_map50": _ms([r["box"]["map50"] for r in runs]),
           "box_map50_95": _ms([r["box"]["map50_95"] for r in runs]),
           "image_accuracy": _ms([r["image_level"]["accuracy"] for r in runs]),
           "image_balanced_accuracy": _ms([r["image_level"]["balanced_accuracy"] for r in runs])}
    for cls in ("pvc", "nylon"):
        out[f"{cls}_precision"] = _ms([r["per_class_seg"][cls]["precision"] for r in runs])
        out[f"{cls}_recall"] = _ms([r["per_class_seg"][cls]["recall"] for r in runs])
    return out


def build() -> dict:
    folds = _group([f"fold_{k}" for k in range(5)])
    matched = _group([f"res_matched_{k}" for k in range(5)])
    cross = {}
    for key in ("cross_a_to_b", "cross_b_to_a"):
        r = _load(key)
        if r:
            cross[key] = {"mask_map50": r["seg"]["map50"], "mask_map50_95": r["seg"]["map50_95"],
                          "image_accuracy": r["image_level"]["accuracy"],
                          "image_balanced_accuracy": r["image_level"]["balanced_accuracy"],
                          "n_val_images": r["n_val_images"], "epochs_run": r["epochs_run"]}
    allcv = _load("all_cv")
    first = _load("fold_0") or {}
    return {
        "schedule": first.get("schedule"), "epochs_budget": first.get("epochs"),
        "patience": first.get("patience"), "batch": first.get("batch"),
        "weights_evaluated": first.get("weights_evaluated"), "device": first.get("meta", {}).get("device"),
        "labels": first.get("labels"), "folds": folds, "resolution_matched": matched,
        "cross_resolution": cross,
        "all_cv": None if not allcv else {
            "epochs_run": allcv["epochs_run"], "mask_map50": allcv["seg"]["map50"],
            "mask_map50_95": allcv["seg"]["map50_95"],
            "note": "validated on its own training images; a fit check, not a generalisation result"},
    }


def _f(x: dict) -> str:
    return f"{x['mean']:.3f} ± {x['std']:.3f}"


def write() -> str:
    s = build()
    (RES / "yolo_summary.json").write_text(json.dumps(s, indent=2), encoding="utf8")
    p2 = json.loads((RES / "phase2_controls.json").read_text(encoding="utf8"))["geometries"]["resample"]
    p5 = json.loads((RES / "phase5_variants.json").read_text(encoding="utf8"))["results"]["baseline"]
    L = ["## YOLOv11 instance segmentation (generated)", "",
         f"Schedule `{s['schedule']}`: up to {s['epochs_budget']} epochs, patience {s['patience']}, batch "
         f"{s['batch']}, device {s['device']}, {s['weights_evaluated']} weights. Cross-validation folds from "
         "`splits.json`; the hold-out was not used.", "",
         "**The labels are the classical segmentation's output (automatic, uncorrected masks), so the model "
         "learns to imitate that segmentation plus the class label; mAP measures agreement with it, not "
         "with human-drawn particle outlines.**", ""]
    f = s["folds"]
    if f:
        L += [f"### {f['n_runs']} cross-validation folds (mean ± std)", "", "| Metric | Value |", "|---|---|",
              f"| Mask mAP50 | {_f(f['mask_map50'])} |", f"| Mask mAP50-95 | {_f(f['mask_map50_95'])} |",
              f"| Box mAP50 | {_f(f['box_map50'])} |", f"| Box mAP50-95 | {_f(f['box_map50_95'])} |",
              f"| PVC precision / recall | {_f(f['pvc_precision'])} / {_f(f['pvc_recall'])} |",
              f"| Nylon precision / recall | {_f(f['nylon_precision'])} / {_f(f['nylon_recall'])} |",
              f"| Image accuracy (aggregated detections) | {_f(f['image_accuracy'])} |",
              f"| Image balanced accuracy | {_f(f['image_balanced_accuracy'])} |", ""]
    if s["resolution_matched"]:
        m = s["resolution_matched"]
        L += ["### Resolution-matched folds (640x480 and 1280x960 only)", "",
              f"Mask mAP50 {_f(m['mask_map50'])}; image accuracy {_f(m['image_accuracy'])}; image balanced "
              f"accuracy {_f(m['image_balanced_accuracy'])}.", ""]
    if s["cross_resolution"]:
        c = s["cross_resolution"]
        L += ["### Cross-resolution control (balanced accuracy of the image decision)", "",
              "| Train to test | YOLO | Per-particle classifier | Metadata shortcut |", "|---|---|---|---|"]
        for key, a, b in (("cross_a_to_b", "a_to_b", "640 to 1280"),
                          ("cross_b_to_a", "b_to_a", "1280 to 640")):
            if key in c:
                short = p2["cross_resolution"][a]["shortcut_size_and_intensity"]["balanced_accuracy"]
                L.append(f"| {b} | {c[key]['image_balanced_accuracy']:.3f} | "
                         f"{p5['cross_resolution'][a]['balanced_accuracy']:.3f} | {short:.3f} |")
        L.append("")
    if s["all_cv"]:
        a = s["all_cv"]
        L += ["### Model on all cross-validation images", "",
              f"{a['epochs_run']} epochs; mask mAP50 {a['mask_map50']:.3f}, mask mAP50-95 "
              f"{a['mask_map50_95']:.3f}, measured on its own training images ({a['note']}).", ""]
    text = "\n".join(L) + "\n"
    (RES / "yolo_summary.md").write_text(text, encoding="utf8")
    return text
