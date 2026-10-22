"""Geometry and photometric normalisation."""
from __future__ import annotations

import cv2
import numpy as np

GEOMETRIES = ("resample", "native")


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img[:, :, ::-1].copy()


def geometry_normalize(
    img: np.ndarray, mode: str = "resample", target: tuple[int, int] = (640, 480)
) -> np.ndarray:
    """`resample`: letterbox to `target` (w, h). `native`: unchanged."""
    if mode == "native":
        return img
    if mode != "resample":
        raise ValueError(f"unknown geometry mode {mode!r}")
    tw, th = target
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp) if (nw, nh) != (w, h) else img
    if (nw, nh) == (tw, th):
        return resized
    out = np.zeros((th, tw, 3), dtype=img.dtype)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def estimate_background(
    img: np.ndarray, kernel: int = 25, downscale: int = 4
) -> np.ndarray:
    """Smooth lower envelope: grayscale opening on a downscaled copy, then upscale.

    `kernel` is in downscaled pixels, so it must exceed the largest particle radius
    (x downscale) or big clumps get absorbed into the background.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // downscale), max(1, h // downscale)),
                       interpolation=cv2.INTER_AREA)
    k = max(3, kernel | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, se)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def subtract_background(
    img: np.ndarray, kernel: int = 25, downscale: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """(background-corrected uint8 image, estimated background)."""
    bg = estimate_background(img, kernel, downscale)
    return cv2.subtract(img, bg), bg


def percentile_normalize(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Scale to ~[0, 1] with one pooled (all channels) percentile pair per image.

    One scale for all channels keeps colour ratios intact while removing the global
    brightness, which differs between classes.
    """
    x = img.astype(np.float32)
    p_lo, p_hi = np.percentile(x, [lo, hi])
    if p_hi - p_lo < 1e-3:
        return np.zeros_like(x)
    return np.clip((x - p_lo) / (p_hi - p_lo), 0, 1)
