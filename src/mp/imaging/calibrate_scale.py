"""Estimate micrometres per pixel from a stage-micrometer image."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import cv2
import numpy as np

from mp.config import config_path, load_config


def line_profile(gray: np.ndarray, line: tuple[int, int, int, int]) -> np.ndarray:
    """Intensity samples (one per pixel of length) along (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = line
    n = max(int(np.hypot(x1 - x0, y1 - y0)), 2)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    return cv2.remap(
        gray.astype(np.float32), xs.astype(np.float32)[None], ys.astype(np.float32)[None],
        cv2.INTER_LINEAR,
    )[0]


def _peak_period(profile: np.ndarray) -> tuple[float, float]:
    """Dominant spatial period in pixels and its peak prominence (peak / median power)."""
    p = profile - profile.mean()
    n = len(p)
    win = np.hanning(n)
    nfft = 1 << (n * 8 - 1).bit_length()
    spec = np.abs(np.fft.rfft(p * win, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft)
    lo = max(1, int(np.searchsorted(freqs, 2.0 / n)))  # at least 2 periods in the line
    k = lo + int(np.argmax(spec[lo:]))
    peak = k
    if 1 <= k < len(spec) - 1:  # parabolic refinement of the peak
        a, b, c = np.log(spec[k - 1] + 1e-12), np.log(spec[k] + 1e-12), np.log(spec[k + 1] + 1e-12)
        denom = a - 2 * b + c
        if denom < 0:
            k = k + 0.5 * (a - c) / denom
    f = np.interp(k, np.arange(len(freqs)), freqs)
    prom = float(spec[peak] / (np.median(spec[lo:]) + 1e-12))
    return 1.0 / f, prom


def estimate_um_per_px(
    image_path: str | Path,
    spacing_um: float,
    line: tuple[int, int, int, int] | None = None,
) -> float:
    """um/px = known division spacing / measured period in pixels.

    Without `line`, horizontal and vertical mean profiles are tried and the one
    with the stronger periodic peak wins.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read {image_path}")
    if line is not None:
        period, _ = _peak_period(line_profile(gray, line))
    else:
        cand = [_peak_period(gray.mean(axis=0)), _peak_period(gray.mean(axis=1))]
        period = max(cand, key=lambda t: t[1])[0]
    return spacing_um / period


def write_um_per_px(value: float, path: Path | None = None) -> None:
    """Set `um_per_px` in configs/data.yaml in place, keeping comments."""
    path = path or config_path()
    text = path.read_text(encoding="utf8")
    new, n = re.subn(r"^um_per_px:.*$", f"um_per_px: {value:.6f}", text, flags=re.M)
    if n != 1:
        raise ValueError("um_per_px key not found exactly once in config")
    path.write_text(new, encoding="utf8")


def require_um_per_px(cfg: dict | None = None) -> float | None:
    """Return calibrated um/px or None, warning loudly. Never invents a value."""
    cfg = cfg or load_config()
    v = cfg.get("um_per_px")
    if v is None:
        warnings.warn(
            "um_per_px is not calibrated; sizes stay in pixels. Run `mp calibrate`.",
            stacklevel=2,
        )
    return v
