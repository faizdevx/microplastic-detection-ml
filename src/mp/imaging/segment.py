"""Classical particle segmentation: threshold + distance-transform watershed."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import watershed


def _threshold(sig: np.ndarray, cfg: dict) -> np.ndarray:
    floor = cfg.get("min_signal", 8)
    method = cfg.get("threshold", "otsu")
    out = np.ones(sig.shape, dtype=bool)
    if method in ("otsu", "both"):
        t, _ = cv2.threshold(sig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out &= sig > max(t, floor)
    if method in ("adaptive", "both"):
        block = cfg.get("adaptive_block", 51) | 1
        ad = cv2.adaptiveThreshold(
            sig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, -cfg.get("adaptive_c", 4),
        )
        out &= (ad > 0) & (sig > floor)
    if method not in ("otsu", "adaptive", "both"):
        raise ValueError(f"unknown threshold method {method!r}")
    return out


def segment(img_bgsub: np.ndarray, cfg: dict) -> np.ndarray:
    """Label image (int32, 0 = background) from a background-corrected RGB uint8 image."""
    ch = cfg.get("channel", 0)
    sig = img_bgsub[:, :, ch] if img_bgsub.ndim == 3 else img_bgsub
    sig = cv2.GaussianBlur(sig, (0, 0), cfg.get("blur_sigma", 1.0))
    binary = _threshold(sig, cfg)

    k = cfg.get("open_kernel", 3)
    if k > 1:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # replicate-pad so the opening does not shave the image border: otherwise
        # border-touching particles would stop touching it and escape the border filter
        pad = k
        padded = cv2.copyMakeBorder(binary.astype(np.uint8), pad, pad, pad, pad,
                                    cv2.BORDER_REPLICATE)
        opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, se)
        binary = opened[pad:-pad, pad:-pad].astype(bool)
    binary = ndi.binary_fill_holes(binary)
    if not binary.any():
        return np.zeros(sig.shape, dtype=np.int32)

    # zero-pad so the image border counts as background: otherwise particles cut by the
    # border get a flat distance plateau and split into spurious pieces
    dist = ndi.distance_transform_edt(np.pad(binary, 1))[1:-1, 1:-1]
    peaks = peak_local_max(
        dist, min_distance=cfg.get("watershed_min_distance", 5),
        labels=cc_label(binary), exclude_border=False,
    )
    markers = np.zeros(sig.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary)

    h, w = labels.shape
    min_area = cfg.get("min_area_px", 20)
    drop_border = cfg.get("drop_border", True)
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for r in regionprops(labels):
        r0, c0, r1, c1 = r.bbox
        touches = r0 == 0 or c0 == 0 or r1 == h or c1 == w
        if r.area >= min_area and not (drop_border and touches):
            keep[r.label] = True
    out = np.where(keep[labels], labels, 0)
    return _relabel(out)


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]


def mask_particles(
    img: np.ndarray, labels: np.ndarray, background: np.ndarray, dilate: int = 3
) -> np.ndarray:
    """Replace every (dilated) particle pixel by the local background estimate."""
    mask = labels > 0
    if dilate > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask.astype(np.uint8), se).astype(bool)
    out = img.copy()
    out[mask] = background[mask]
    return out
