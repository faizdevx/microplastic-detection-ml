"""Per-particle and background feature extraction."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops
from tqdm import tqdm

from mp.config import load_config, load_named, resolve
from mp.imaging import normalize as nz
from mp.imaging import segment as sg

EPS = 1e-3
CH = ("R", "G", "B")
SHAPE_COLS = (
    ["area_px", "eq_diameter_px", "perimeter_px", "circularity", "solidity", "convexity",
     "aspect_ratio", "eccentricity", "major_axis_px", "minor_axis_px"]
    + [f"hu{i}" for i in range(1, 8)]
)
RATIO_COLS = ["ratio_R_over_G", "frac_R", "frac_G", "frac_B"]
NORM_COLS = [f"norm_{s}_{c}" for c in CH for s in ("mean", "std", "max")]
RAW_COLS = [f"raw_{s}_{c}" for c in CH for s in ("mean", "std", "max")] + [
    "contrast_R", "contrast_ratio_R"]
BG_COLS = (
    [f"bg_{s}_{c}" for c in CH for s in ("mean", "std", "p5", "p50", "p95")]
    + ["bg_row_streak", "bg_laplacian", "bg_illum_lr", "bg_illum_tb"]
)
FEATURE_GROUPS = {
    "shape": SHAPE_COLS, "ratio": RATIO_COLS,
    "intensity_norm": NORM_COLS, "intensity_raw": RAW_COLS,
}


def feature_columns(groups: list[str]) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def _convexity(sub_mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(
        np.pad(sub_mask, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return np.nan
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    hull_per = cv2.arcLength(cv2.convexHull(c), True)
    return hull_per / per if per > 0 else np.nan


def particle_features(
    labels: np.ndarray, raw: np.ndarray, norm: np.ndarray, ring_px: int = 5
) -> pd.DataFrame:
    """One row per labelled particle. `raw` uint8 background-corrected, `norm` float [0,1]."""
    fg = labels > 0
    h, w = labels.shape
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    rawf = raw.astype(np.float32) / 255.0
    rows = []
    for r in regionprops(labels):
        r0, c0, r1, c1 = r.bbox
        r0p, c0p = max(0, r0 - ring_px - 1), max(0, c0 - ring_px - 1)
        r1p, c1p = min(h, r1 + ring_px + 1), min(w, c1 + ring_px + 1)
        sub = labels[r0p:r1p, c0p:c1p] == r.label
        ring = cv2.dilate(sub.astype(np.uint8), se).astype(bool) & ~fg[r0p:r1p, c0p:c1p]
        per = max(r.perimeter, 1e-6)
        row = {
            "label": r.label,
            "cy": r.centroid[0], "cx": r.centroid[1],
            "area_px": r.area,
            "eq_diameter_px": r.equivalent_diameter_area,
            "perimeter_px": r.perimeter,
            "circularity": min(4 * np.pi * r.area / per**2, 1.5),
            "solidity": r.solidity,
            "convexity": _convexity(r.image),
            "aspect_ratio": r.axis_major_length / max(r.axis_minor_length, 1e-6),
            "eccentricity": r.eccentricity,
            "major_axis_px": r.axis_major_length,
            "minor_axis_px": r.axis_minor_length,
        }
        hu = r.moments_hu
        for i in range(7):
            row[f"hu{i + 1}"] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30)
        rawm = rawf[r0p:r1p, c0p:c1p][sub]
        normm = norm[r0p:r1p, c0p:c1p][sub]
        for k, c in enumerate(CH):
            row[f"raw_mean_{c}"] = rawm[:, k].mean()
            row[f"raw_std_{c}"] = rawm[:, k].std()
            row[f"raw_max_{c}"] = rawm[:, k].max()
            row[f"norm_mean_{c}"] = normm[:, k].mean()
            row[f"norm_std_{c}"] = normm[:, k].std()
            row[f"norm_max_{c}"] = normm[:, k].max()
        m = np.array([row[f"raw_mean_{c}"] for c in CH])
        tot = m.sum() + 3 * EPS
        row["ratio_R_over_G"] = (m[0] + EPS) / (m[1] + EPS)
        row["frac_R"], row["frac_G"], row["frac_B"] = (m + EPS) / tot
        if ring.any():
            ring_r = rawf[r0p:r1p, c0p:c1p, 0][ring].mean()
            row["contrast_R"] = m[0] - ring_r
            row["contrast_ratio_R"] = (m[0] + EPS) / (ring_r + EPS)
        else:
            row["contrast_R"] = np.nan
            row["contrast_ratio_R"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def background_features(img: np.ndarray, labels: np.ndarray, dilate: int = 3) -> dict:
    """Statistics of the geometry-normalised image OUTSIDE the (dilated) particles."""
    mask = labels > 0
    if dilate > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask.astype(np.uint8), se).astype(bool)
    bgm = ~mask
    out = {}
    if bgm.sum() < 100:
        return {c: np.nan for c in BG_COLS}
    for k, c in enumerate(CH):
        v = img[:, :, k][bgm].astype(np.float32)
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        out.update({f"bg_mean_{c}": v.mean(), f"bg_std_{c}": v.std(),
                    f"bg_p5_{c}": p5, f"bg_p50_{c}": p50, f"bg_p95_{c}": p95})
    gray = img.astype(np.float32).mean(axis=2)
    rowmask = bgm.sum(axis=1) > 0
    rm = np.array([gray[i][bgm[i]].mean() for i in np.where(rowmask)[0]])
    out["bg_row_streak"] = float(rm.std() / (rm.mean() + EPS))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    out["bg_laplacian"] = float(lap[bgm].mean())
    h, w = gray.shape
    left, right = gray[:, : w // 2][bgm[:, : w // 2]], gray[:, w // 2 :][bgm[:, w // 2 :]]
    top, bot = gray[: h // 2][bgm[: h // 2]], gray[h // 2 :][bgm[h // 2 :]]
    out["bg_illum_lr"] = float((left.mean() - right.mean()) / (gray[bgm].mean() + EPS))
    out["bg_illum_tb"] = float((top.mean() - bot.mean()) / (gray[bgm].mean() + EPS))
    return out


def prepare(path, geometry: str, pre: dict) -> dict:
    """Load and normalise one image. Returns the arrays every later step needs."""
    rgb = nz.load_rgb(path)
    geo = nz.geometry_normalize(rgb, geometry, tuple(pre["target_size"]))
    bgsub, bg = nz.subtract_background(geo, pre["bg_open_kernel"], pre["bg_downscale"])
    norm = nz.percentile_normalize(bgsub, pre["pct_lo"], pre["pct_hi"])
    return {"geo": geo, "bgsub": bgsub, "bg": bg, "norm": norm}


def process_image(path, geometry: str, ccfg: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    a = prepare(path, geometry, ccfg["preprocess"])
    labels = sg.segment(a["bgsub"], ccfg["segmentation"])
    parts = particle_features(labels, a["bgsub"], a["norm"])
    bgf = background_features(a["geo"], labels, ccfg["segmentation"]["mask_dilate"])
    bgf["n_particles"] = int(labels.max())
    bgf["fg_fraction"] = float((labels > 0).mean())
    return parts, bgf, labels


def _worker(rel: str, root: str, geometry: str, ccfg: dict):
    parts, bgf, _ = process_image(f"{root}/{rel}", geometry, ccfg)
    parts.insert(0, "rel_path", rel)
    bgf["rel_path"] = rel
    return parts, bgf


def particles_path(cfg: dict, geometry: str):
    return resolve(cfg["paths"]["groups"]).parent / f"particles_{geometry}.csv.gz"


def images_path(cfg: dict, geometry: str):
    return resolve(cfg["paths"]["groups"]).parent / f"bgfeatures_{geometry}.csv"


def build_tables(geometry: str, cfg: dict | None = None, ccfg: dict | None = None,
                 n_jobs: int = -1, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features for every unique image (CV pool and hold-out) and cache to disk."""
    from mp.data.dataset import load_pool

    cfg = cfg or load_config()
    ccfg = ccfg or load_named("classifier")
    pool = load_pool(cfg)
    rels = pool.images["rel_path"].tolist()
    if limit:
        rels = rels[:limit]
    root = str(resolve(cfg["paths"]["extract_dir"]))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(r, root, geometry, ccfg)
        for r in tqdm(rels, desc=f"features[{geometry}]", unit="img")
    )
    parts = pd.concat([p for p, _ in res if len(p)], ignore_index=True)
    bg = pd.DataFrame([b for _, b in res])
    if not limit:
        parts.to_csv(particles_path(cfg, geometry), index=False)
        bg.to_csv(images_path(cfg, geometry), index=False)
    return parts, bg
