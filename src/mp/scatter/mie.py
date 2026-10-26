"""Mie scattering by homogeneous spheres.

Efficiencies come from `miepython`. The angular amplitudes S1, S2 are computed here
with a numpy recursion over the Mie coefficients (Bohren & Huffman normalisation),
because that is much faster than miepython's per-angle routine at x of several hundred.
"""
from __future__ import annotations

from functools import lru_cache

import miepython
import numpy as np


def size_parameter(d_um, wavelength_nm: float, n_medium: float):
    """x = pi d n_medium / lambda0 (diameter and wavelength in the same length unit)."""
    return np.pi * np.asarray(d_um, float) * 1e-6 * n_medium / (wavelength_nm * 1e-9)


def relative_index(n_particle: float, k_particle: float, n_medium: float) -> complex:
    """m = (n - ik) / n_medium (miepython convention: negative imaginary part)."""
    return complex(n_particle / n_medium, -k_particle / n_medium)


@lru_cache(maxsize=4096)
def _coefficients(m_re: float, m_im: float, x: float):
    an, bn = miepython.coefficients(complex(m_re, m_im), x)
    return np.asarray(an), np.asarray(bn)


def amplitudes(m: complex, x: float, theta_deg) -> tuple[np.ndarray, np.ndarray]:
    """Complex scattering amplitudes (S1, S2) at the scattering angles `theta_deg`."""
    mu = np.cos(np.radians(np.atleast_1d(np.asarray(theta_deg, float))))
    an, bn = _coefficients(round(m.real, 12), round(m.imag, 12), round(float(x), 9))
    s1 = np.zeros(mu.shape, dtype=complex)
    s2 = np.zeros(mu.shape, dtype=complex)
    pi_prev, pi_cur = np.zeros_like(mu), np.ones_like(mu)
    for n in range(1, len(an) + 1):
        tau = n * mu * pi_cur - (n + 1) * pi_prev
        c = (2 * n + 1) / (n * (n + 1))
        s1 += c * (an[n - 1] * pi_cur + bn[n - 1] * tau)
        s2 += c * (an[n - 1] * tau + bn[n - 1] * pi_cur)
        pi_prev, pi_cur = pi_cur, ((2 * n + 1) * mu * pi_cur - (n + 1) * pi_prev) / n
    return s1, s2


def angular_intensity(m: complex, x: float, theta_deg) -> dict[str, np.ndarray]:
    """|S1|^2 (perpendicular), |S2|^2 (parallel) and the unpolarised mean.

    The differential cross section per particle is unpolarised / k^2 with
    k = 2 pi n_medium / lambda0.
    """
    s1, s2 = amplitudes(m, x, theta_deg)
    i_perp, i_par = np.abs(s1) ** 2, np.abs(s2) ** 2
    return {"perp": i_perp, "par": i_par, "unpolarised": 0.5 * (i_perp + i_par)}


@lru_cache(maxsize=4096)
def _efficiencies(m_re: float, m_im: float, x: float):
    qext, qsca, _qback, g = miepython.single_sphere(complex(m_re, m_im), x, 0, False)
    return float(qext), float(qsca), float(g)


def cross_sections(m: complex, x: float) -> dict[str, float]:
    """Q_ext, Q_sca and asymmetry parameter g."""
    qext, qsca, g = _efficiencies(round(m.real, 12), round(m.imag, 12), round(float(x), 9))
    return {"q_ext": qext, "q_sca": qsca, "g": g}


def phase_function(m: complex, x: float, theta_deg) -> np.ndarray:
    """Normalised phase function p(theta) per steradian, integrating to 1 over the sphere."""
    q = cross_sections(m, x)["q_sca"]
    return angular_intensity(m, x, theta_deg)["unpolarised"] / (np.pi * q * x**2)
