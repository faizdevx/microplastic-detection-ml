"""YOLOv11 instance segmentation: per-fold dataset files, training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mp.annotate.auto import stem_of, yolo_root
from mp.config import load_config, load_named, resolve
from mp.data.dataset import Pool, load_pool
from mp.eval.metrics import point_metrics, run_meta

HIDDEN_RES = ("640x480", "1280x960")


def _img_path(rel: str) -> Path:
    cls_dir, stem = stem_of(rel)
    return yolo_root() / "images" / cls_dir / f"{stem}.jpg"


def _write_list(path: Path, rels: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [_img_path(r) for r in rels if _img_path(r).exists()]
    path.write_text("\n".join(p.resolve().as_posix() for p in existing) + "\n", encoding="utf8")
    return path.resolve().as_posix()


def _write_yaml(path: Path, train: str, val: str, names: list[str]) -> Path:
    path.write_text(
        yaml.safe_dump({"path": yolo_root().resolve().as_posix(), "train": train,
                        "val": val, "names": dict(enumerate(names))}, sort_keys=False),
        encoding="utf8",
    )
    return path


def build_datasets(pool: Pool | None = None, ycfg: dict | None = None) -> dict[str, Path]:
    """Dataset YAMLs from splits.json. Ultralytics never sees a split it chose itself.

    fold_k          train = CV images outside fold k,   val = fold k
    all_cv          train = every CV image,             val = same CV images (no hold-out
                    image is used for training, validation or epoch selection)
    res_matched_k   same as fold_k restricted to resolutions present in both classes
    cross_a_to_b    train on 640x480 CV images, val on 1280x960 CV images (and reverse)
    holdout.txt     hold-out image list, used only by the final evaluation
    """
    pool = pool or load_pool()
    ycfg = ycfg or load_named("yolo_seg")
    names = ycfg["classes"]
    root = yolo_root()
    lists = root / "lists"
    out: dict[str, Path] = {}
    cv = pool.cv()
    res = cv["width"].astype(str) + "x" + cv["height"].astype(str)

    for k in range(pool.n_folds):
        tr, va = pool.fold_split(k)
        out[f"fold_{k}"] = _write_yaml(
            root / f"fold_{k}.yaml",
            _write_list(lists / f"fold_{k}_train.txt", tr["rel_path"].tolist()),
            _write_list(lists / f"fold_{k}_val.txt", va["rel_path"].tolist()), names)
        matched = set(HIDDEN_RES)
        trm = tr[(tr["width"].astype(str) + "x" + tr["height"].astype(str)).isin(matched)]
        vam = va[(va["width"].astype(str) + "x" + va["height"].astype(str)).isin(matched)]
        out[f"res_matched_{k}"] = _write_yaml(
            root / f"res_matched_{k}.yaml",
            _write_list(lists / f"res_matched_{k}_train.txt", trm["rel_path"].tolist()),
            _write_list(lists / f"res_matched_{k}_val.txt", vam["rel_path"].tolist()), names)

    all_list = _write_list(lists / "all_cv.txt", cv["rel_path"].tolist())
    out["all_cv"] = _write_yaml(root / "all_cv.yaml", all_list, all_list, names)

    for tag, (a, b) in {"cross_a_to_b": HIDDEN_RES, "cross_b_to_a": HIDDEN_RES[::-1]}.items():
        te = cv[res == b]
        tr = cv[(res == a) & ~cv["group_id"].isin(te["group_id"])]
        out[tag] = _write_yaml(
            root / f"{tag}.yaml",
            _write_list(lists / f"{tag}_train.txt", tr["rel_path"].tolist()),
            _write_list(lists / f"{tag}_val.txt", te["rel_path"].tolist()), names)

    _write_list(lists / "holdout.txt", pool.holdout()["rel_path"].tolist())
    return out


def build_smoke(pool: Pool, ycfg: dict) -> Path:
    n = ycfg["smoke"]["n_images"]
    tr, va = pool.fold_split(0)
    half = n // 2

    def balanced(df, m):
        return pd.concat([df[df["y"] == c].head(m // 2) for c in (0, 1)])

    root = yolo_root() / "lists"
    t = _write_list(root / "smoke_train.txt", balanced(tr, half)["rel_path"].tolist())
    v = _write_list(root / "smoke_val.txt", balanced(va, n - half)["rel_path"].tolist())
    return _write_yaml(yolo_root() / "smoke.yaml", t, v, ycfg["classes"])


def pick_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_args(ycfg: dict, data: Path, name: str, smoke: bool) -> dict:
    a = ycfg["augment"]
    args = dict(
        data=str(data), imgsz=ycfg["imgsz"], epochs=ycfg["epochs"], batch=ycfg["batch"],
        patience=ycfg["patience"], optimizer=ycfg["optimizer"], lr0=ycfg["lr0"],
        seed=ycfg["seed"], deterministic=ycfg["deterministic"], workers=ycfg["workers"],
        project=str((yolo_root() / "runs").resolve()), name=name, exist_ok=True,
        device=pick_device(), plots=False, **a,
    )
    if smoke:
        s = ycfg["smoke"]
        args.update(imgsz=s["imgsz"], epochs=s["epochs"], batch=s["batch"], workers=0,
                    device="cpu", close_mosaic=0, patience=ycfg["patience"])
    return args


def image_class_scores(model, rels: list[str], imgsz: int, device: str) -> pd.DataFrame:
    """P(NYLON) per image = confidence mass of nylon masks / all masks."""
    rows = []
    for rel in rels:
        p = _img_path(rel)
        r = model.predict(str(p), imgsz=imgsz, device=device, verbose=False, conf=0.25)[0]
        if r.boxes is None or len(r.boxes) == 0:
            rows.append({"rel_path": rel, "prob": np.nan, "n_pred": 0})
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        tot = conf.sum()
        rows.append({"rel_path": rel, "prob": float(conf[cls == 1].sum() / tot), "n_pred": len(cls)})
    return pd.DataFrame(rows)


def counting_error(pred: np.ndarray, true: np.ndarray) -> dict:
    d = pred - true
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return {"mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "loa_low": float(d.mean() - 1.96 * sd), "loa_high": float(d.mean() + 1.96 * sd)}


def label_provenance() -> dict:
    p = yolo_root() / "label_sources.csv"
    if not p.exists():
        return {"corrected": 0, "auto": 0, "description": "no labels exported"}
    d = pd.read_csv(p)
    n_cor = int((d["label_source"] == "corrected").sum())
    return {
        "corrected": n_cor, "auto": int((d["label_source"] == "auto").sum()),
        "description": "trained on auto-generated masks (uncorrected)" if n_cor == 0 else
        f"{n_cor} images use human-corrected masks, rest auto-generated",
    }


def train_and_eval(data_key: str, smoke: bool = False, epochs: int | None = None,
                   cfg: dict | None = None) -> dict:
    """Train on data_key's train list, evaluate on its val list, save results JSON."""
    from ultralytics import YOLO

    cfg = cfg or load_config()
    ycfg = load_named("yolo_seg")
    pool = load_pool(cfg)
    if smoke:
        data = build_smoke(pool, ycfg)
        name = "smoke"
    else:
        data = build_datasets(pool, ycfg)[data_key]
        name = data_key
    args = train_args(ycfg, data, name, smoke)
    if epochs:
        args["epochs"] = epochs
    weights = ycfg["smoke"]["model"] if smoke else ycfg["model"]
    model = YOLO(weights)
    model.train(**args)
    best = Path(model.trainer.best) if getattr(model.trainer, "best", None) else None
    ev_model = YOLO(str(best)) if best and best.exists() else model
    m = ev_model.val(data=str(data), imgsz=args["imgsz"], device=args["device"],
                     plots=False, workers=0, verbose=False)
    val_list = Path(yaml.safe_load(Path(data).read_text())["val"])
    rels = []
    for line in val_list.read_text().splitlines():
        if line.strip():
            p = Path(line)
            rels.append(f"{p.parent.name}/{p.stem}")
    full = pool.images.assign(key=lambda d: d["rel_path"].map(lambda r: f"{stem_of(r)[0]}/{stem_of(r)[1]}"))
    lookup = dict(zip(full["key"], full["rel_path"]))
    val_rels = [lookup[r] for r in rels if r in lookup]
    sc = image_class_scores(ev_model, val_rels, args["imgsz"], args["device"])
    meta_df = pool.images[["rel_path", "y", "group_id"]]
    sc = sc.merge(meta_df, on="rel_path")
    ok = sc.dropna(subset=["prob"])
    res = {
        "meta": run_meta(args["device"]),
        "data": data_key if not smoke else "smoke",
        "weights": weights,
        "epochs": args["epochs"],
        "labels": label_provenance(),
        "seg": {"map50": float(m.seg.map50), "map50_95": float(m.seg.map)},
        "box": {"map50": float(m.box.map50), "map50_95": float(m.box.map)},
        "per_class_seg": {
            ycfg["classes"][i]: {"precision": float(m.seg.p[i]), "recall": float(m.seg.r[i])}
            for i in range(min(len(ycfg["classes"]), len(m.seg.p)))
        },
        "image_level": point_metrics(ok["y"].to_numpy(), ok["prob"].to_numpy()) if len(ok) else None,
        "n_val_images": len(sc), "n_val_no_prediction": int(sc["prob"].isna().sum()),
        "counting_error": None,
        "counting_error_note": "needs human-corrected masks; none imported"
        if label_provenance()["corrected"] == 0 else "see annotation_quality.json",
    }
    out = resolve(cfg["paths"]["results"]) / f"phase2_yolo_{res['data']}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf8")
    sc.to_csv(yolo_root() / f"preds_{res['data']}.csv", index=False)
    return res


def gpu_status() -> dict:
    """Is a CUDA device usable by PyTorch? Writes nothing; used for results/phase5_yolo_status.json."""
    import shutil
    import subprocess

    try:
        import torch

        cuda, build = bool(torch.cuda.is_available()), torch.__version__
    except Exception:
        cuda, build = False, "not installed"
    names = []
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True)
        names = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return {"torch_cuda_available": cuda, "torch_build": build, "nvidia_gpus_detected": names}


def write_status(cfg: dict | None = None) -> dict:
    """Train only with a usable GPU; otherwise record why not and what to run."""
    cfg = cfg or load_config()
    g = gpu_status()
    prov = label_provenance()
    out = {
        "meta": run_meta("cpu"),
        "gpu": g,
        "trained": False,
        "reason": ("a CUDA-enabled PyTorch build and a GPU are required; none is usable here"
                   if not g["torch_cuda_available"] else "GPU available: run the commands below"),
        "labels": prov,
        "commands": [
            "python -m mp.cli segment",
            "for k in 0 1 2 3 4: python -m mp.cli train-seg --fold $k",
            "for k in 0 1 2 3 4: python -m mp.cli train-seg --data res_matched_$k",
            "python -m mp.cli train-seg --data cross_a_to_b",
            "python -m mp.cli train-seg --data cross_b_to_a",
            "python -m mp.cli train-seg --all",
        ],
        "note": ("masks are the classical segmentation's output, so a trained model learns to "
                 "imitate it plus the class label"),
    }
    (resolve(cfg["paths"]["results"]) / "phase5_yolo_status.json").write_text(
        json.dumps(out, indent=2), encoding="utf8")
    return out
