"""Auto-mask bootstrap, human-correction export/import, and mask quality (IoU)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mp.config import load_config, load_named, repo_root, resolve
from mp.data.dataset import HOLDOUT, load_pool
from mp.imaging import features as F
from mp.imaging.segment import segment

CLASSES = ["pvc", "nylon"]  # YOLO class ids 0 / 1 (match configs/yolo_seg.yaml)


def yolo_root() -> Path:
    return repo_root() / "data" / "yolo"


def ann_root() -> Path:
    return repo_root() / "data" / "annotations"


def stem_of(rel: str) -> tuple[str, str]:
    p = Path(rel)
    return p.parent.name, p.stem


def label_to_polygons(labels: np.ndarray, eps: float = 0.8) -> list[np.ndarray]:
    """One simplified polygon (Nx2, pixels) per labelled particle."""
    polys = []
    for i in range(1, int(labels.max()) + 1):
        m = (labels == i).astype(np.uint8)
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(c) >= 3:
            polys.append(c.astype(np.float32))
    return polys


def write_yolo_txt(path: Path, polys: list[np.ndarray], cls_id: int, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in polys:
        xy = (p / np.array([w, h], dtype=np.float32)).clip(0, 1).reshape(-1)
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in xy))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")


def read_yolo_txt(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        xy = np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * np.array([w, h], np.float32)
        out.append((int(parts[0]), xy))
    return out


def export_auto(cfg: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    """Segment every unique image (resampled 640x480 view) and write YOLO-seg labels."""
    cfg = cfg or load_config()
    ccfg = load_named("classifier")
    pool = load_pool(cfg)
    imgs = pool.images if not limit else pool.images.head(limit)
    root = resolve(cfg["paths"]["extract_dir"])
    rows = []
    for _, r in tqdm(imgs.iterrows(), total=len(imgs), desc="auto labels", unit="img"):
        a = F.prepare(root / r["rel_path"], "resample", ccfg["preprocess"])
        labels = segment(a["bgsub"], ccfg["segmentation"])
        h, w = labels.shape
        cls_dir, stem = stem_of(r["rel_path"])
        polys = label_to_polygons(labels)
        write_yolo_txt(ann_root() / "auto" / cls_dir / f"{stem}.txt", polys, int(r["y"]), w, h)
        img_out = yolo_root() / "images" / cls_dir / f"{stem}.jpg"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out), a["geo"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"rel_path": r["rel_path"], "n_auto_particles": len(polys), "w": w, "h": h})
    out = pd.DataFrame(rows)
    sync_yolo_labels(out["rel_path"].tolist())
    return out


def sync_yolo_labels(rels: list[str]) -> pd.DataFrame:
    """data/yolo/labels = corrected label if it exists, else auto. Records provenance."""
    rows = []
    for rel in rels:
        cls_dir, stem = stem_of(rel)
        cor = ann_root() / "corrected" / cls_dir / f"{stem}.txt"
        auto = ann_root() / "auto" / cls_dir / f"{stem}.txt"
        src, kind = (cor, "corrected") if cor.exists() else (auto, "auto")
        if not src.exists():
            continue
        dst = yolo_root() / "labels" / cls_dir / f"{stem}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append({"rel_path": rel, "label_source": kind})
    df = pd.DataFrame(rows)
    yolo_root().mkdir(parents=True, exist_ok=True)
    df.to_csv(yolo_root() / "label_sources.csv", index=False)
    return df


def export_correction_set(n_per_class: int = 150, seed: int = 42, cfg: dict | None = None) -> Path:
    """COCO json + images for CVAT / Label Studio: one representative per group."""
    cfg = cfg or load_config()
    pool = load_pool(cfg)
    reps = pool.images.drop_duplicates("group_id")
    pick = (
        reps.groupby("class", group_keys=False)
        .apply(lambda d: d.sample(min(n_per_class, len(d)), random_state=seed))
        .sort_values("rel_path")
    )
    out_dir = ann_root() / "to_correct"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    ann_id = 1
    for img_id, (_, r) in enumerate(pick.iterrows(), start=1):
        cls_dir, stem = stem_of(r["rel_path"])
        src = yolo_root() / "images" / cls_dir / f"{stem}.jpg"
        if not src.exists():
            continue
        dst = out_dir / "images" / cls_dir / f"{stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        h, w = cv2.imread(str(src)).shape[:2]
        images.append({
            "id": img_id, "file_name": f"{cls_dir}/{stem}.jpg", "width": w, "height": h,
            "rel_path": r["rel_path"],
            "split": "holdout" if r["fold"] == HOLDOUT else f"cv_fold_{int(r['fold'])}",
        })
        for cid, poly in read_yolo_txt(ann_root() / "auto" / cls_dir / f"{stem}.txt", w, h):
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            anns.append({
                "id": ann_id, "image_id": img_id, "category_id": cid + 1, "iscrowd": 0,
                "segmentation": [poly.reshape(-1).round(1).tolist()],
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": float(cv2.contourArea(poly)),
            })
            ann_id += 1
    coco = {
        "images": images, "annotations": anns,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
        "info": {"description": "auto-generated masks to be corrected by a human"},
    }
    path = out_dir / "coco_auto.json"
    path.write_text(json.dumps(coco), encoding="utf8")
    return path


def _raster(polys, h: int, w: int) -> list[np.ndarray]:
    out = []
    for p in polys:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        out.append(m.astype(bool))
    return out


def mask_iou(auto_polys, corr_polys, h: int, w: int) -> dict:
    """Semantic IoU (union masks) and instance matching of corrected vs auto masks."""
    a, c = _raster(auto_polys, h, w), _raster(corr_polys, h, w)
    ua = np.any(a, axis=0) if a else np.zeros((h, w), bool)
    uc = np.any(c, axis=0) if c else np.zeros((h, w), bool)
    union = (ua | uc).sum()
    sem = float((ua & uc).sum() / union) if union else 1.0
    best = []
    for mc in c:
        ious = [(mc & ma).sum() / max((mc | ma).sum(), 1) for ma in a]
        best.append(max(ious) if ious else 0.0)
    return {
        "semantic_iou": sem,
        "instance_best_ious": best,
        "n_auto": len(a), "n_corrected": len(c),
    }


def import_corrected(coco_path: str | Path, cfg: dict | None = None) -> dict:
    """Corrected COCO -> YOLO labels (data/annotations/corrected) + auto-vs-corrected IoU."""
    cfg = cfg or load_config()
    coco = json.loads(Path(coco_path).read_text(encoding="utf8"))
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    sem, inst, n_img = [], [], 0
    rels = []
    for im in coco["images"]:
        cls_dir, stem = im["file_name"].split("/")[-2], Path(im["file_name"]).stem
        w, h = im["width"], im["height"]
        corr = []
        for a in by_img.get(im["id"], []):
            for seg in a["segmentation"]:
                poly = np.array(seg, np.float32).reshape(-1, 2)
                if len(poly) >= 3:
                    corr.append((a["category_id"] - 1, poly))
        write_path = ann_root() / "corrected" / cls_dir / f"{stem}.txt"
        write_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{cid} " + " ".join(
                f"{v:.6f}" for v in (p / np.array([w, h], np.float32)).clip(0, 1).reshape(-1))
            for cid, p in corr
        ]
        write_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
        auto = [p for _, p in read_yolo_txt(ann_root() / "auto" / cls_dir / f"{stem}.txt", w, h)]
        m = mask_iou(auto, [p for _, p in corr], h, w)
        sem.append(m["semantic_iou"])
        inst.extend(m["instance_best_ious"])
        n_img += 1
        rels.append(im.get("rel_path") or f"{cls_dir.upper()}/{stem}.jpg")
    sync_yolo_labels(rels)
    inst = np.array(inst) if inst else np.array([])
    res = {
        "n_corrected_images": n_img,
        "semantic_iou_mean": float(np.mean(sem)) if sem else None,
        "instance_best_iou_mean": float(inst.mean()) if len(inst) else None,
        "instance_recall_at_iou50": float((inst >= 0.5).mean()) if len(inst) else None,
    }
    out = resolve(cfg["paths"]["results"]) / "annotation_quality.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf8")
    return res
