"""Spectral slope γ(R) and γ-for-snapshot helpers.

Used by the two-parameter running-coupling fit ``β_nl(σ², γ)``: at a
given Lagrangian smoothing radius ``R``, ``γ = d ln σ²(R) / d ln R`` is
the local slope of the smoothed variance, a dimensionless proxy for the
spectral tilt at that scale. It differentiates the cutoff and no-cutoff
PM suites even at matched ``σ²_J``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from discodj import DiscoDJ


__all__ = [
    "smoothed_sigma2",
    "spectral_gamma",
    "R_from_sigma2",
    "gamma_for_snapshot",
    "reconstruct_pk",
]


def _top_hat_window(kR):
    kR = np.asarray(kR)
    # Guard against kR → 0 (analytic limit W = 1 at kR = 0).
    small = kR < 1e-3
    x = np.where(small, 1e-3, kR)
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.where(small, 1.0, W)


def smoothed_sigma2(k: np.ndarray, Pk: np.ndarray, R: float) -> float:
    r"""Top-hat-smoothed variance

    .. math::  \sigma^2(R) = \frac{1}{2\pi^2} \int k^2 P(k) |W_\text{TH}(kR)|^2 \, dk

    with ``W_TH(x) = 3(\sin x - x\cos x)/x^3``.
    """
    k = np.asarray(k); Pk = np.asarray(Pk)
    W = _top_hat_window(k * R)
    integrand = k**2 * Pk * W**2
    return float(np.trapezoid(integrand, k) / (2.0 * np.pi**2))


def spectral_gamma(k: np.ndarray, Pk: np.ndarray, R: float,
                   dln: float = 0.05) -> float:
    r"""Local slope ``γ = d ln σ²(R) / d ln R`` via centred finite difference
    in ``ln R``.  ``dln`` is the half-step in ``ln R``.
    """
    lnR = np.log(R)
    s_plus = smoothed_sigma2(k, Pk, np.exp(lnR + dln))
    s_minus = smoothed_sigma2(k, Pk, np.exp(lnR - dln))
    # Guard against zeros (possible in heavily-cut-off regime)
    if s_plus <= 0 or s_minus <= 0:
        return float("nan")
    return float((np.log(s_plus) - np.log(s_minus)) / (2.0 * dln))


def R_from_sigma2(k: np.ndarray, Pk: np.ndarray, sigma2_target: float,
                  R_bracket: tuple[float, float] = (1e-2, 1e3),
                  tol: float = 1e-3, max_iter: int = 60) -> float:
    """Invert ``σ²(R) = σ²_target`` for ``R`` by bisection.

    Returns the Lagrangian radius (in the same units as ``1/k``, i.e. Mpc/h
    if ``k`` is in h/Mpc) at which the top-hat-smoothed variance equals
    ``sigma2_target``. Assumes ``σ²(R)`` is monotonically decreasing with R
    (true for any reasonable cosmological spectrum).
    """
    lo, hi = R_bracket
    s_lo = smoothed_sigma2(k, Pk, lo)
    s_hi = smoothed_sigma2(k, Pk, hi)
    if not (s_hi < sigma2_target < s_lo):
        # Out of bracket — return the closer endpoint.
        return lo if abs(s_lo - sigma2_target) < abs(s_hi - sigma2_target) else hi
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)           # geometric midpoint
        s_mid = smoothed_sigma2(k, Pk, mid)
        if abs(s_mid - sigma2_target) / sigma2_target < tol:
            return float(mid)
        if s_mid > sigma2_target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def reconstruct_pk(snap) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the P(k) table used by a PM snapshot's ICs.

    Uses ``snap.meta["cosmo"]`` (if present — nonstandard cosmology runs)
    or the default Planck18EEBAOSN cosmology, and applies the Gaussian
    cutoff ``exp(-(k/k_cut)²)`` if the snap's meta reports
    ``cutoff_mode == "cutoff"``.

    Returns ``(k, Pk_at_z=1)`` — the linear spectrum at a=1 in DiscoDJ's
    internal normalisation. Note: we only ever use *ratios* and local
    slopes of Pk in the spectral-slope computation, so the absolute
    normalisation is irrelevant.
    """
    cosmo = snap.meta.get("cosmo", None)
    dj = DiscoDJ(
        dim=3, res=max(snap.meta.get("N", 64), 64), boxsize=snap.meta.get("L", 250.0),
        cosmo=cosmo if cosmo is not None else "Planck18EEBAOSN",
    ).with_timetables().with_linear_ps()
    k = np.asarray(dj._pk_table["k"])
    Pk = np.asarray(dj._pk_table["Pk"])
    if snap.meta.get("cutoff_mode") == "cutoff":
        k_cut = snap.meta.get("k_cut", 0.0)
        if k_cut > 0:
            Pk = Pk * np.exp(-(k / k_cut) ** 2)
    return k, Pk


def gamma_for_snapshot(snap, sigma2_trG: float | None = None,
                        dln: float = 0.1) -> tuple[float, float, float]:
    """Return ``(σ²_trG, R_eff, γ)`` for a single snapshot.

    Rebuilds the snapshot's linear ``P(k)``, finds the Lagrangian radius
    ``R_eff`` at which ``σ²_lin(R) = σ²_trG`` (the cascade's matched σ²),
    then evaluates ``γ(R_eff)``.

    If ``sigma2_trG`` is passed it's used directly; otherwise the caller
    must compute it (``gamma_for_snapshot`` does not differentiate the
    positions, to avoid re-running the Fourier machinery).
    """
    if sigma2_trG is None:
        raise ValueError("Pass sigma2_trG (from deformation_gradient_jacobian) — "
                         "gamma_for_snapshot does not recompute it.")
    k, Pk = reconstruct_pk(snap)
    R_eff = R_from_sigma2(k, Pk, sigma2_trG)
    gamma = spectral_gamma(k, Pk, R_eff, dln=dln)
    return float(sigma2_trG), float(R_eff), float(gamma)
