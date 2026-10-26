"""Photon Monte Carlo for multiple scattering in a square cuvette.

Geometry: box 0 <= x <= L, |y|, |z| <= L/2. The beam enters at the origin along +x.
Detectors sit in the xy-plane at angle theta from +x. At every interaction a
next-event estimate is added for each detector (phase function x solid angle x
transmission to the wall). Forced collisions (truncated exponential free paths) keep
the variance low at the small optical depths of this problem.
"""
from __future__ import annotations

import numpy as np


def _exit_distance(pos: np.ndarray, d: np.ndarray, length: float) -> np.ndarray:
    """Distance from `pos` along direction `d` to the cuvette wall."""
    big = np.inf
    lo = np.array([0.0, -length / 2, -length / 2])
    hi = np.array([length, length / 2, length / 2])
    out = np.full(len(pos), big)
    for ax in range(3):
        dc = d[:, ax] if d.ndim == 2 else np.full(len(pos), d[ax])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dc > 1e-12, (hi[ax] - pos[:, ax]) / dc,
                         np.where(dc < -1e-12, (lo[ax] - pos[:, ax]) / dc, big))
        out = np.minimum(out, np.maximum(t, 0.0))
    return out


def _sample_directions(u: np.ndarray, cos_t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    sin_t = np.sqrt(np.clip(1 - cos_t**2, 0, None))
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    near = np.abs(uz) > 0.99999
    denom = np.sqrt(np.clip(1 - uz**2, 1e-30, None))
    cp, sp = np.cos(phi), np.sin(phi)
    nx = np.where(near, sin_t * cp, sin_t * (ux * uz * cp - uy * sp) / denom + ux * cos_t)
    ny = np.where(near, sin_t * sp, sin_t * (uy * uz * cp + ux * sp) / denom + uy * cos_t)
    nz = np.where(near, np.sign(uz) * cos_t, -sin_t * cp * denom + uz * cos_t)
    out = np.stack([nx, ny, nz], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def detector_power(n_per_m3: float, popt, rig, phase, cfg: dict, seed: int) -> np.ndarray:
    theta_grid, p_theta = phase
    n_ph = int(cfg.get("photons", 8000))
    max_order = int(cfg.get("max_order", 8))
    mu = n_per_m3 * popt.sigma_ext
    albedo = popt.sigma_sca / popt.sigma_ext
    rng = np.random.default_rng(seed)
    length = rig.path_m
    th_rad = np.radians(theta_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p_theta[1:] * np.sin(th_rad[1:])
                                                  + p_theta[:-1] * np.sin(th_rad[:-1]))
                                           * np.diff(th_rad))])
    cdf /= cdf[-1]
    offsets = np.linspace(-rig.half_angle_deg, rig.half_angle_deg, 5)
    det_dirs = np.array([[np.cos(np.radians(t)), np.sin(np.radians(t)), 0.0]
                         for t in rig.theta_deg])
    pos = np.zeros((n_ph, 3))
    direc = np.tile([1.0, 0.0, 0.0], (n_ph, 1))
    w = np.ones(n_ph)
    power = np.zeros(len(det_dirs))
    for _ in range(max_order):
        s_max = _exit_distance(pos, direc, length)
        p_int = 1.0 - np.exp(-mu * s_max)
        u = rng.random(n_ph)
        s = -np.log1p(-u * p_int) / max(mu, 1e-30)
        w = w * p_int * albedo
        pos = pos + direc * s[:, None]
        for i, dd in enumerate(det_dirs):
            cos_g = np.clip(direc @ dd, -1.0, 1.0)
            gamma = np.degrees(np.arccos(cos_g))
            # average over the detector acceptance, as the single-scattering formula does
            ph = np.mean([np.interp(np.clip(gamma + o, 0.0, 180.0), theta_grid, p_theta)
                          for o in offsets], axis=0)
            trans = np.exp(-mu * _exit_distance(pos, dd, length))
            power[i] += np.sum(w * ph * rig.solid_angle * trans)
        cos_t = np.cos(np.interp(rng.random(n_ph), cdf, th_rad))
        direc = _sample_directions(direc, cos_t, 2 * np.pi * rng.random(n_ph))
    return rig.laser_W * power / n_ph
