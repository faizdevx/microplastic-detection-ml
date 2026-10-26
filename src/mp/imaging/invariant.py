"""Features that depend less on exposure and background, plus the helpers the controls need.

V1  local-normalisation features (`V1_FEATURES`)
V2  exposure randomisation (`augment_array`): the Phase 2 features, trained on gain/offset/gamma copies
V3  size-normalised, background-masked particle patches (`make_patches`)
Pseudo-particles (`pseudo_labels`) are discs of class-independent size placed on background pixels. A feature
set that still predicts the class from them is reading the background.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops
from tqdm import tqdm

from mp.config import load_config, load_named, repo_root, resolve
from mp.imaging import features as F
from mp.imaging import normalize as nz
from mp.imaging import segment as sg

EPS = 1e-3
CH = ("R", "G", "B")
V1_EXTRA = (
    [f"v1_rel_{c}" for c in CH] + [f"v1_surr_{c}" for c in CH] + [f"v1_cv_{c}" for c in CH]
    + ["v1_rg", "v1_gb", "v1_lap", "v1_edge"]
)
V1_FEATURES = F.SHAPE_COLS + F.RATIO_COLS + V1_EXTRA
BASE_FEATURES = F.feature_columns(list(F.FEATURE_GROUPS))
PATCH = 48
PATCH_FIT = 40
MAX_PATCHES = 40
N_AUG = 3


def data_dir():
    d = repo_root() / "data" / "phase5"
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_for(rel: str, salt: int = 0) -> int:
    return (zlib.crc32(rel.encode()) + 7919 * salt) % (2**32)


# ------------------------------------------------------------------ preparation --------
def prepare_array(geo: np.ndarray, pre: dict) -> dict:
    bgsub, bg = nz.subtract_background(geo, pre["bg_open_kernel"], pre["bg_downscale"])
    norm = nz.percentile_normalize(bgsub, pre["pct_lo"], pre["pct_hi"])
    return {"geo": geo, "bgsub": bgsub, "bg": bg, "norm": norm}


def augment_array(geo: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Random exposure change: gamma, gain and additive offset (uint8 in, uint8 out)."""
    gain = float(rng.uniform(0.5, 2.0))
    gamma = float(rng.uniform(0.7, 1.4))
    offset = float(rng.uniform(-8.0, 16.0))
    x = (geo.astype(np.float32) / 255.0) ** gamma * 255.0 * gain + offset
    return np.clip(x, 0, 255).astype(np.uint8), {"gain": gain, "gamma": gamma, "offset": offset}


def image_p99(labels: np.ndarray, raw: np.ndarray) -> float:
    """99th percentile of luminance over all particle pixels of the image."""
    lum = raw.astype(np.float32).mean(axis=2) / 255.0
    v = lum[labels > 0]
    return float(np.percentile(v, 99)) if v.size else 1.0


# ---------------------------------------------------------------------- V1 -------------
def invariant_features(labels: np.ndarray, raw: np.ndarray, p99: float | None = None,
                       exclude: np.ndarray | None = None, ring_px: int = 5) -> pd.DataFrame:
    """Per-particle features that use only ratios to the surround, the image's own particle
    brightness scale, and texture. No absolute intensity is returned."""
    fg = labels > 0
    if exclude is not None:
        fg = fg | exclude
    p99 = p99 if p99 is not None else image_p99(labels, raw)
    rawf = raw.astype(np.float32) / 255.0
    lum = rawf.mean(axis=2)
    lap = cv2.Laplacian(lum, cv2.CV_32F)
    gx, gy = cv2.Sobel(lum, cv2.CV_32F, 1, 0), cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    h, w = labels.shape
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    for r in regionprops(labels):
        r0, c0, r1, c1 = r.bbox
        a, b = max(0, r0 - ring_px - 1), max(0, c0 - ring_px - 1)
        c, d = min(h, r1 + ring_px + 1), min(w, c1 + ring_px + 1)
        sub = labels[a:c, b:d] == r.label
        ring = cv2.dilate(sub.astype(np.uint8), se).astype(bool) & ~fg[a:c, b:d]
        inside = rawf[a:c, b:d][sub]
        m = inside.mean(axis=0)
        row = {"label": r.label}
        for k, ch in enumerate(CH):
            row[f"v1_rel_{ch}"] = m[k] / (p99 + EPS)
            row[f"v1_cv_{ch}"] = inside[:, k].std() / (m[k] + EPS)
            row[f"v1_surr_{ch}"] = (
                (m[k] + EPS) / (rawf[a:c, b:d][ring][:, k].mean() + EPS) if ring.any() else np.nan)
        row["v1_rg"] = (m[0] + EPS) / (m[1] + EPS)
        row["v1_gb"] = (m[1] + EPS) / (m[2] + EPS)
        ml = float(lum[a:c, b:d][sub].mean())
        row["v1_lap"] = float(lap[a:c, b:d][sub].var() / (ml**2 + EPS))
        edge = sub & ~cv2.erode(sub.astype(np.uint8), ero).astype(bool)
        row["v1_edge"] = float(grad[a:c, b:d][edge].mean() / (ml + EPS)) if edge.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pseudo-particles ---
def pseudo_labels(labels: np.ndarray, rng: np.random.Generator, margin: int = 4,
                  tries: int = 40, radius: tuple[int, int] = (4, 14)) -> np.ndarray:
    """Background-only control regions: one disc per real particle, placed on pixels that touch
    no particle. Radii come from one fixed range, so shape and size carry no class information
    and anything a classifier learns from these regions comes from the background pixels."""
    h, w = labels.shape
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    forbidden = cv2.dilate((labels > 0).astype(np.uint8), se).astype(bool)
    out = np.zeros_like(labels)
    placed = 0
    for _ in range(int(labels.max())):
        r = int(rng.integers(radius[0], radius[1] + 1))
        gy, gx = np.mgrid[-r:r + 1, -r:r + 1]
        keep = gy**2 + gx**2 <= r * r
        dy, dx = gy[keep], gx[keep]
        lo, hi_y, hi_x = r + margin, h - r - margin, w - r - margin
        if hi_y <= lo or hi_x <= lo:
            continue
        for _try in range(tries):
            cy, cx = int(rng.integers(lo, hi_y)), int(rng.integers(lo, hi_x))
            if forbidden[cy + dy, cx + dx].any():
                continue
            placed += 1
            out[cy + dy, cx + dx] = placed
            forbidden[max(0, cy - r - margin):cy + r + margin + 1,
                      max(0, cx - r - margin):cx + r + margin + 1] = True
            break
    return out


# --------------------------------------------------------------------------- V3 --------
def make_patch(labels: np.ndarray, raw: np.ndarray, region) -> np.ndarray:
    """48x48 uint8 patch: particle pixels only, size-normalised, percentile-normalised."""
    r0, c0, r1, c1 = region.bbox
    a, b = max(0, r0 - 2), max(0, c0 - 2)
    c, d = min(labels.shape[0], r1 + 2), min(labels.shape[1], c1 + 2)
    mask = (labels[a:c, b:d] == region.label)
    crop = raw[a:c, b:d].copy()
    crop[~mask] = 0
    scale = PATCH_FIT / max(crop.shape[:2])
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                      interpolation=interp)
    nz_pix = crop[crop.any(axis=2)]
    p99 = float(np.percentile(nz_pix, 99)) if nz_pix.size else 1.0
    crop = np.clip(crop.astype(np.float32) * (230.0 / max(p99, 1.0)), 0, 255).astype(np.uint8)
    out = np.zeros((PATCH, PATCH, 3), np.uint8)
    y0, x0 = (PATCH - crop.shape[0]) // 2, (PATCH - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out


def make_patches(labels: np.ndarray, raw: np.ndarray, rng: np.random.Generator,
                 cap: int = MAX_PATCHES) -> tuple[np.ndarray, list[int]]:
    regs = regionprops(labels)
    if len(regs) > cap:
        regs = [regs[i] for i in sorted(rng.choice(len(regs), cap, replace=False))]
    if not regs:
        return np.zeros((0, PATCH, PATCH, 3), np.uint8), []
    return np.stack([make_patch(labels, raw, r) for r in regs]), [r.label for r in regs]


# ---------------------------------------------------------------------- tables ---------
def _all_features(labels, a, p99=None, exclude=None) -> pd.DataFrame:
    base = F.particle_features(labels, a["bgsub"], a["norm"], exclude=exclude)
    if not len(base):
        return base
    inv = invariant_features(labels, a["bgsub"], p99=p99, exclude=exclude)
    return base.merge(inv, on="label")


def process_image(path, rel: str, ccfg: dict, n_aug: int = N_AUG) -> dict:
    pre, scfg = ccfg["preprocess"], ccfg["segmentation"]
    geo = nz.geometry_normalize(nz.load_rgb(path), "resample", tuple(pre["target_size"]))
    a = prepare_array(geo, pre)
    labels = sg.segment(a["bgsub"], scfg)
    rng = np.random.default_rng(seed_for(rel))
    real = _all_features(labels, a)
    p99 = image_p99(labels, a["bgsub"])
    pl = pseudo_labels(labels, rng)
    fg = labels > 0
    pseudo = _all_features(pl, a, p99=p99, exclude=fg)
    pa, ids_a = make_patches(labels, a["bgsub"], rng)
    pb, ids_b = make_patches(pl, a["bgsub"], rng)
    aug = []
    for k in range(n_aug):
        g2, _ = augment_array(geo, np.random.default_rng(seed_for(rel, k + 1)))
        a2 = prepare_array(g2, pre)
        l2 = sg.segment(a2["bgsub"], scfg)
        f2 = _all_features(l2, a2)
        if len(f2):
            f2["aug_k"] = k
            aug.append(f2)
    for df in (real, pseudo):
        if len(df):
            df.insert(0, "rel_path", rel)
    for df in aug:
        df.insert(0, "rel_path", rel)
    return {"real": real, "pseudo": pseudo, "aug": aug, "patches_real": pa, "ids_real": ids_a,
            "patches_pseudo": pb, "ids_pseudo": ids_b, "rel": rel}


def build_tables(n_jobs: int = -1, limit: int | None = None) -> dict:
    """Feature, pseudo-particle, augmentation and patch tables for the CV pool only.
    The hold-out is never read."""
    from mp.data.dataset import load_pool

    cfg, ccfg = load_config(), load_named("classifier")
    pool = load_pool(cfg)
    rels = pool.cv()["rel_path"].tolist()[:limit]
    root = str(resolve(cfg["paths"]["extract_dir"]))
    res = Parallel(n_jobs=n_jobs)(delayed(process_image)(f"{root}/{r}", r, ccfg)
                                  for r in tqdm(rels, desc="phase5 tables", unit="img"))
    out = data_dir()
    real = pd.concat([r["real"] for r in res if len(r["real"])], ignore_index=True)
    pseudo = pd.concat([r["pseudo"] for r in res if len(r["pseudo"])], ignore_index=True)
    aug = pd.concat([d for r in res for d in r["aug"]], ignore_index=True)
    real.to_csv(out / "real.csv.gz", index=False)
    pseudo.to_csv(out / "pseudo.csv.gz", index=False)
    aug.to_csv(out / "aug.csv.gz", index=False)
    for kind in ("real", "pseudo"):
        arr = np.concatenate([r[f"patches_{kind}"] for r in res if len(r[f"patches_{kind}"])])
        idx = pd.DataFrame([(r["rel"], lab) for r in res for lab in r[f"ids_{kind}"]],
                           columns=["rel_path", "label"])
        np.save(out / f"patches_{kind}.npy", arr)
        idx.to_csv(out / f"patches_{kind}_idx.csv", index=False)
    return {"images": len(rels), "real": len(real), "pseudo": len(pseudo), "aug": len(aug)}
