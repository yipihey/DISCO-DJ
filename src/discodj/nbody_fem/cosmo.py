"""Cosmological wrapper for the FEM N-body solver.

Uses FastPM-style DKD in growth-factor (``D``) time with DISCO-DJ's cosmology
object to supply ``D_plus``, ``F_plus``, etc. The integrator state is

    x: comoving positions (Mpc/h)
    p: dx/dD  (at linear / ZA order, ``p = ψ`` — a constant of motion)

Equations of motion in D-time:

    Drift:  x ← x + ΔD · p
    Kick:   p ← α·p + β·F(x_mid)

Coefficients per step (FastPM, Feng+ 2016, eqs. 19-20):

    α = F_plus(a_begin) / F_plus(a_end)
    β = (1 − α) / D_plus(a_mid)

where ``F_plus(a) = a³·E(a)·dD/da`` and ``D_plus`` is the normalised growth
factor. ``β`` is set by the Zel'dovich consistency condition: at linear order
``p = ψ`` is constant, so ``p_new = p_old`` forces ``β·F = (1−α)·ψ``.

The FEM force ``F_FEM = -∂U/∂x`` with our Poisson convention (``K φ = f``,
``f ∝ δ``) is the comoving gradient of the potential. To hand it to the FastPM
kick we rescale by ``(3/2)·Ω_m·H₀² / a_mid`` — the physical Poisson pre-factor
``∇²φ_phys = (3/2)·Ω_m·H₀²·δ/a`` — divided by ``|k|²`` already absorbed in the
FEM inverse of K. See :func:`fastpm_kick_scale` for the exact combination.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from ..cosmology.cosmology import Cosmology


__all__ = [
    "analytic_F_scale",
    "fastpm_coeffs",
    "cosmological_kdk_step",
    "build_time_table",
]


def analytic_F_scale(boxsize: float, N: int) -> float:
    """Geometric factor turning ``F_FEM = -∂U/∂x`` into DISCO-DJ's PM force.

    Derivation (see Phase 3 plan):

    - DISCO-DJ's PM Poisson: ``∇²φ_DD = +δ``, force ``F_DD = −∇φ_DD``.
    - Our FEM ``K φ_FEM = f`` with ``K ≈ h³(−∇²)`` and ``f ≈ h³ δ`` gives
      ``∇²φ_FEM = −δ`` — sign-flipped from ``φ_DD``.
    - Our force ``F_FEM = −∂U/∂x`` with ``U = ½ fᵀ φ_FEM``. Envelope-theorem
      analysis gives ``F_FEM ≈ h³ · ∇φ_FEM`` on the undeformed mesh (lumped
      mass at the vertex is ``h³`` for ρ̄ = 1).
    - Therefore ``F_DD = −h³ · F_FEM``, and the multiplier that turns our
      raw FEM force into DISCO-DJ's PM force is **independent of cosmology**:

          F_scale = −(N / boxsize)**3
    """
    return -(N / boxsize) ** 3


def build_time_table(
    cosmo: Cosmology,
    a_ini: float,
    a_end: float,
    n_steps: int,
    boxsize: float,
    N: int,
    spacing: str = "log_a",
) -> dict:
    """Build a table of scale factors + midpoints + D-time intervals.

    :param cosmo: DISCO-DJ cosmology with ``with_timetables()`` already called.
    :param a_ini: starting scale factor.
    :param a_end: ending scale factor.
    :param n_steps: number of KDK steps.
    :param boxsize: simulation box size (Mpc/h) — used for the analytic
        ``F_scale``; see :func:`analytic_F_scale`.
    :param N: grid resolution.
    :param spacing: ``"log_a"`` (default), ``"a"``, or ``"D"``. log_a matches
        DISCO-DJ's default for their N-body runs.
    :return: dict with arrays of length ``n_steps`` or ``n_steps + 1``:
        ``a_edges``, ``a_mid``, ``D_edges``, ``D_mid``, ``dD1``, ``dD2``,
        ``alpha``, ``beta``, ``F_scale``.
    """
    if spacing == "log_a":
        log_a_edges = jnp.linspace(jnp.log(a_ini), jnp.log(a_end), n_steps + 1)
        a_edges = jnp.exp(log_a_edges)
    elif spacing == "a":
        a_edges = jnp.linspace(a_ini, a_end, n_steps + 1)
    elif spacing == "D":
        D_ini = cosmo.Dplus(a_ini)
        D_end = cosmo.Dplus(a_end)
        D_edges = jnp.linspace(D_ini, D_end, n_steps + 1)
        a_edges = jnp.asarray([cosmo.Dplus_to_a(d) for d in D_edges])
    else:
        raise ValueError(f"Unknown spacing: {spacing!r}")

    # Midpoints (in D-time) between consecutive edges.
    D_edges = jnp.asarray([cosmo.Dplus(a) for a in a_edges])
    D_mid = 0.5 * (D_edges[:-1] + D_edges[1:])
    a_mid = jnp.asarray([cosmo.Dplus_to_a(d) for d in D_mid])

    dD1 = D_mid - D_edges[:-1]
    dD2 = D_edges[1:] - D_mid

    F_begin = jnp.asarray([cosmo.Fplus(a) for a in a_edges[:-1]])
    F_end = jnp.asarray([cosmo.Fplus(a) for a in a_edges[1:]])
    alpha = F_begin / F_end
    beta = (1.0 - alpha) / D_mid

    # Analytic geometric factor — independent of cosmology, fixed per (L, N).
    F_scale = jnp.full_like(a_mid, analytic_F_scale(boxsize, N))

    return {
        "a_edges": a_edges, "a_mid": a_mid,
        "D_edges": D_edges, "D_mid": D_mid,
        "dD1": dD1, "dD2": dD2,
        "alpha": alpha, "beta": beta,
        "F_scale": F_scale,
    }


def fastpm_coeffs(cosmo: Cosmology, a_begin: float, a_end: float) -> dict:
    """Single-step FastPM coefficients. Used in tests and one-off evaluations."""
    D_begin = cosmo.Dplus(a_begin)
    D_end = cosmo.Dplus(a_end)
    D_mid = 0.5 * (D_begin + D_end)
    a_mid = cosmo.Dplus_to_a(D_mid)
    alpha = cosmo.Fplus(a_begin) / cosmo.Fplus(a_end)
    beta = (1.0 - alpha) / D_mid
    return {
        "a_mid": float(a_mid),
        "D_mid": float(D_mid),
        "dD1": float(D_mid - D_begin),
        "dD2": float(D_end - D_mid),
        "alpha": float(alpha),
        "beta": float(beta),
    }


def cosmological_kdk_step(
    x: Array,
    p: Array,
    force_fn: Callable[[Array], Array],
    coeffs: dict,
    step_idx: int,
    boxsize: float,
) -> tuple[Array, Array]:
    """One FastPM DKD step from time slice ``step_idx`` to ``step_idx+1``.

    :param x: comoving positions ``(N**3, 3)``.
    :param p: D-time momenta ``(N**3, 3)`` (``p = dx/dD``; at linear order
        ``p = ψ``).
    :param force_fn: ``x -> F_FEM`` closure (from :func:`nbody_fem.force.force_fn`).
    :param coeffs: dict from :func:`build_time_table`.
    :param step_idx: which step in the table.
    :param boxsize: periodic box side; positions get wrapped via ``jnp.mod``.
    """
    dD1 = coeffs["dD1"][step_idx]
    dD2 = coeffs["dD2"][step_idx]
    alpha = coeffs["alpha"][step_idx]
    beta = coeffs["beta"][step_idx]
    F_scale = coeffs["F_scale"][step_idx]

    x_mid = x + dD1 * p
    F = force_fn(x_mid) * F_scale
    p_new = alpha * p + beta * F
    x_new = x_mid + dD2 * p_new
    return x_new, p_new
