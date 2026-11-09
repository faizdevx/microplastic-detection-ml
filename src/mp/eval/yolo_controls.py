"""YOLOv11-seg confound controls and the paired comparison with the per-particle classifier.

background discs  real particles inpainted away, one disc of one fixed colour per removed particle
                  placed on background pixels (same placement rule as invariant.pseudo_labels); the
                  fold's model decides PVC/Nylon from what is left, which is the background
normalised        per-image, per-channel 1-99 percentile stretch before training and testing, on the
                  resolution-matched folds
comparison        exact McNemar test and a group bootstrap of the accuracy difference, image level
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mp.annotate.auto import ann_root, read_yolo_txt, stem_of, yolo_root
from mp.config import load_named, repo_root, resolve
from mp.data.dataset import load_pool
from mp.eval.controls import acc_bal
from mp.eval.metrics import run_meta
from mp.imaging.invariant import pseudo_labels

SEED = 7
N_FOLDS = 5


def _res(name: str) -> Path:
    return resolve("results") / name


def norm_root() -> Path:
    return repo_root() / "data" / "yolo_norm"


def disc_root(k: int) -> Path:
    return repo_root() / "data" / "yolo_discs" / f"fold_{k}"


def normalise(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img.reshape(-1, 3), [1, 99], axis=0)
    return ((img.astype(np.float32) - lo) / np.maximum(hi - lo, 1) * 255).clip(0, 255).astype(np.uint8)


def build_norm_tree() -> Path:
    src, dst = yolo_root(), norm_root()
    for p in (src / "images").rglob("*.jpg"):
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), normalise(cv2.imread(str(p))), [cv2.IMWRITE_JPEG_QUALITY, 95])
    shutil.copytree(src / "labels", dst / "labels", dirs_exist_ok=True)
    return dst


def _label_image(rel: str, h: int, w: int) -> np.ndarray:
    cls_dir, stem = stem_of(rel)
    lab = np.zeros((h, w), np.int32)
    for i, (_, poly) in enumerate(read_yolo_txt(ann_root() / "auto" / cls_dir / f"{stem}.txt", w, h), 1):
        cv2.fillPoly(lab, [np.round(poly).astype(np.int32)], i)
    return lab


def disc_image(img: np.ndarray, lab: np.ndarray, colour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = cv2.dilate((lab > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    out = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    out[pseudo_labels(lab, rng) > 0] = colour
    return out


def build_disc_tree(k: int, rels: list[str]) -> Path:
    """Disc images for fold k's validation images. The disc colour is the median particle colour over
    all of them, both classes together, so it carries no class information."""
    src = yolo_root() / "images"
    items = []
    for rel in rels:
        cls_dir, stem = stem_of(rel)
        p = src / cls_dir / f"{stem}.jpg"
        if p.exists():
            img = cv2.imread(str(p))
            items.append((cls_dir, stem, img, _label_image(rel, *img.shape[:2])))
    colour = np.median(np.concatenate([img[lab > 0] for _, _, img, lab in items]), axis=0).astype(np.uint8)
    rng = np.random.default_rng(SEED + k)
    root = disc_root(k)
    for cls_dir, stem, img, lab in items:
        out = root / "images" / cls_dir / f"{stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), disc_image(img, lab, colour, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return root


def train(ks: list[int] | None = None, workers: int | None = None) -> None:
    """Retrain the fold models (their weights are needed for the disc control) and the normalised runs."""
    from mp.models import yolo

    if not norm_root().exists():
        build_norm_tree()
    for k in ks if ks is not None else range(N_FOLDS):
        if not _res(f"phase7_yolo_fold_{k}.json").exists():
            yolo.train_and_eval(f"fold_{k}", schedule="short", tag=f"fold_{k}", workers=workers)
        if not _res(f"phase7_yolo_norm_res_matched_{k}.json").exists():
            yolo.train_and_eval(f"res_matched_{k}", schedule="short", root=norm_root(),
                                tag=f"norm_res_matched_{k}", workers=workers)


def _bal(preds: pd.DataFrame) -> dict:
    ok = preds.dropna(subset=["prob"])
    return {**acc_bal(ok["y"].to_numpy(), ok["prob"].to_numpy()), "n": int(len(preds)),
            "n_no_prediction": int(preds["prob"].isna().sum())}


def discs() -> dict:
    from ultralytics import YOLO

    from mp.models import yolo

    pool = load_pool()
    ycfg = load_named("yolo_seg")
    folds, orig = [], []
    for k in range(N_FOLDS):
        run = json.loads(_res(f"phase7_yolo_fold_{k}.json").read_text(encoding="utf8"))
        _, va = pool.fold_split(k)
        rels = va["rel_path"].tolist()
        root = build_disc_tree(k, rels)
        model = YOLO(run["weights_path"])
        sc = yolo.image_class_scores(model, rels, ycfg["imgsz"], yolo.pick_device(), root)
        sc = sc.merge(pool.images[["rel_path", "y"]], on="rel_path")
        folds.append({"fold": k, **_bal(sc)})
        orig.append({"fold": k, **run["image_level"]})
    return {"per_fold": folds, "original_images_per_fold": orig}


def _mean_std(vals: list[float]) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _runs(pattern: str) -> list[dict]:
    return [json.loads(_res(pattern.format(k)).read_text(encoding="utf8")) for k in range(N_FOLDS)]


def _bal_runs(runs: list[dict]) -> dict:
    return _mean_std([r["image_level"]["balanced_accuracy"] for r in runs])


def _map_runs(runs: list[dict]) -> dict:
    return _mean_std([r["seg"]["map50"] for r in runs])


def write_controls() -> dict:
    d = discs()
    retr, stored = _runs("phase7_yolo_fold_{}.json"), _runs("phase2_yolo_fold_{}.json")
    matched, norm = _runs("phase2_yolo_res_matched_{}.json"), _runs("phase7_yolo_norm_res_matched_{}.json")
    base = json.loads(_res("phase5_variants.json").read_text(encoding="utf8"))["results"]["baseline"]
    out = {
        "meta": run_meta(),
        "retrained_folds": {
            "note": "fold models retrained with the same short schedule because the phase 2 weights were "
                    "not kept; the disc control uses these weights",
            "mask_map50": _map_runs(retr), "image_balanced_accuracy": _bal_runs(retr),
            "stored_mask_map50": _map_runs(stored), "stored_image_balanced_accuracy": _bal_runs(stored),
        },
        "background_discs": {
            **d,
            "balanced_accuracy": _mean_std([f["balanced_accuracy"] for f in d["per_fold"]]),
            "classifier_balanced_accuracy": base["pseudo_particles"]["balanced_accuracy"],
        },
        "normalised_resolution_matched": {
            "per_fold": [{"fold": k, "mask_map50": r["seg"]["map50"], **r["image_level"]}
                         for k, r in enumerate(norm)],
            "balanced_accuracy": _bal_runs(norm), "mask_map50": _map_runs(norm),
            "unnormalised_balanced_accuracy": _bal_runs(matched),
            "classifier_balanced_accuracy": base["resolution_matched"]["balanced_accuracy"],
        },
    }
    _res("phase7_yolo_controls.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    return out


# ------------------------------------------------------------------------- review masks ---
def _coco_labels(coco: dict) -> dict[str, list[str]]:
    """file_name -> YOLO-seg label lines from a COCO file."""
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = {}
    for im in coco["images"]:
        wh = np.array([im["width"], im["height"]], np.float32)
        out[im["file_name"]] = [
            f"{a['category_id'] - 1} " + " ".join(
                f"{v:.6f}" for v in (np.array(s, np.float32).reshape(-1, 2) / wh).clip(0, 1).reshape(-1))
            for a in by_img.get(im["id"], []) for s in a["segmentation"] if len(s) >= 6]
    return out


def review_map(coco_path: str | Path) -> dict:
    """Mask mAP of each fold model on its own review images, against the corrected masks and against the
    automatic masks of the same images. Needs the retrained fold weights (phase7_yolo_fold_k.json)."""
    from ultralytics import YOLO

    from mp.models import yolo

    coco = json.loads(Path(coco_path).read_text(encoding="utf8"))
    corrected = _coco_labels(coco)
    ycfg = load_named("yolo_seg")
    out = {}
    for k in range(N_FOLDS):
        files = [im["file_name"] for im in coco["images"] if im.get("split") == f"cv_fold_{k}"]
        if not files:
            continue
        run = json.loads(_res(f"phase7_yolo_fold_{k}.json").read_text(encoding="utf8"))
        model = YOLO(run["weights_path"])
        for kind in ("corrected", "auto"):
            root = repo_root() / "data" / "yolo_review" / kind / f"fold_{k}"
            for f in files:
                img = root / "images" / f
                img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(yolo_root() / "images" / f, img)
                lab = (root / "labels" / f).with_suffix(".txt")
                lab.parent.mkdir(parents=True, exist_ok=True)
                if kind == "corrected":
                    lab.write_text("\n".join(corrected[f]) + "\n", encoding="utf8")
                else:
                    shutil.copyfile((yolo_root() / "labels" / f).with_suffix(".txt"), lab)
            lst = yolo._write_list(root / "lists" / "val.txt", files)
            data = yolo._write_yaml(root / "review.yaml", lst, lst, ycfg["classes"])
            m = model.val(data=str(data), imgsz=ycfg["imgsz"], device=yolo.pick_device(), plots=False,
                          workers=0, verbose=False)
            out.setdefault(kind, []).append({"fold": k, "n_images": len(files),
                                             "mask_map50": float(m.seg.map50),
                                             "mask_map50_95": float(m.seg.map)})
    return out


def write_review(coco_path: str | Path) -> dict:
    """Auto vs corrected masks (IoU, counting error) and YOLO mAP on corrected masks.
    Writes results/review_quality.json."""
    from mp.annotate.auto import qa_report

    qa = qa_report(coco_path)
    res = {"meta": run_meta(), "qa": {k: v for k, v in qa.items() if k != "per_image"},
           "yolo_mask_map": review_map(coco_path)}
    _res("review_quality.json").write_text(json.dumps(res, indent=2), encoding="utf8")
    return res


# --------------------------------------------------------------------------- comparison ---
def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired correct/incorrect outcomes."""
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"only_first_correct": b, "only_second_correct": c, "p_value": p}


def group_bootstrap(df: pd.DataFrame, n: int = 2000, seed: int = SEED) -> dict:
    """95% interval of mean(yolo_ok) - mean(cls_ok), resampling image groups."""
    g = df.groupby("group_id")[["yolo_ok", "cls_ok"]].agg(["sum", "count"])
    s = (g[("yolo_ok", "sum")] - g[("cls_ok", "sum")]).to_numpy()
    m = g[("yolo_ok", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    diffs = s[idx].sum(1) / m[idx].sum(1)
    return {"difference": float(s.sum() / m.sum()), "ci95": [float(np.percentile(diffs, 2.5)),
                                                             float(np.percentile(diffs, 97.5))],
            "n_bootstrap": n, "n_groups": int(len(s))}


def classifier_oof() -> pd.DataFrame:
    from mp.eval import phase5

    rcfg = phase5.load_robust_config()
    data = phase5.load_data(with_patches=False)
    return phase5.run_cv(phase5.make_variants(rcfg)["baseline"], data)


def write_comparison() -> dict:
    cls = classifier_oof()
    yp = pd.concat([pd.read_csv(yolo_root() / f"preds_fold_{k}.csv") for k in range(N_FOLDS)])
    df = cls[["rel_path", "y", "group_id", "prob"]].merge(
        yp[["rel_path", "prob"]].rename(columns={"prob": "yolo_prob"}), on="rel_path")
    n_all = len(df)
    df = df.dropna(subset=["prob", "yolo_prob"])
    df["cls_ok"] = (df["prob"] >= 0.5).astype(int) == df["y"]
    df["yolo_ok"] = (df["yolo_prob"] >= 0.5).astype(int) == df["y"]
    stored = json.loads(_res("phase5_variants.json").read_text(encoding="utf8"))
    out = {
        "meta": run_meta(),
        "first": "YOLOv11-seg (retrained fold models)", "second": "per-particle classifier (baseline)",
        "n_paired_images": int(len(df)), "n_dropped_no_prediction": int(n_all - len(df)),
        "yolo_accuracy": float(df["yolo_ok"].mean()), "classifier_accuracy": float(df["cls_ok"].mean()),
        "classifier_cv_accuracy_stored": stored["results"]["baseline"]["cv"]["accuracy"],
        "mcnemar": mcnemar(df["yolo_ok"].to_numpy(), df["cls_ok"].to_numpy()),
        "accuracy_difference": group_bootstrap(df),
    }
    _res("phase7_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    return out
