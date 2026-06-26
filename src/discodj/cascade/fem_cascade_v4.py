"""Cascade v4: CLPT-style increment with 3LPT cubic term.

Extends v3 by adding a 3LPT-structured cubic piece to the Gaussian
increment. The full non-Gaussian increment is

    dG_nonlin = β · ∇∇⁻²[δ₁² − G₁:G₁]        (2LPT, v3)
              + γ · ∇∇⁻²[δ₁³ − 3 δ₁ · G₁:G₁]  (3LPT scalar, new)

The 3LPT coefficient ``γ = 5/21`` reproduces Bernardeau's single-mode
F₃ kernel at its symmetric leading term. The second term inside the
3LPT bracket is the shear-coupled cubic ``δ·s²`` contribution — for
an isotropic Gaussian 1LPT field it averages to zero but locally
(per realisation) it carries real physics and reshapes the tails.

As with β, the running of γ with σ² is the key question: canonical
Bernardeau ``γ = 5/21`` is a UV value; at the IR of the cascade it
may be renormalised to a different — possibly negative — value.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from ..core.grids import get_fourier_grid
from ..fem.assembly import (
    _edges_from_G, _element_contributions_from_edges,
    assemble_stiffness_stencil,
)
from ..fem.hessian import fourier_hessian
from ..fem.kuhn import kuhn_connectivity, lagrangian_edge_vectors
from ..fem.solve import solve_poisson
from .fem_cascade import one_lpt_increment


__all__ = ["fem_cascade_v4", "CascadeV4Result"]


@dataclass
class CascadeV4Result:
    J: Array
    G: Array
    delta: Array
    sigma2_trG: float


# ---------------------------------------------------------------- source ----


def _source_delta(G_flat: Array, use_abs_J: bool, alpha_nl: float) -> Array:
    J = jnp.linalg.det(jnp.eye(3) + G_flat)
    J_eff = jnp.abs(J) if use_abs_J else J
    d = 1.0 / J_eff - 1.0
    return d + alpha_nl * d * d


def _assemble_source(G, boxsize: float, use_abs_J: bool,
                     alpha_nl: float) -> Array:
    N = G.shape[0]
    G_flat = G.reshape(N**3, 3, 3)
    conn = jnp.asarray(kuhn_connectivity(N))
    lag_edges = lagrangian_edge_vectors(N, boxsize)
    edges_def = _edges_from_G(G_flat, conn, lag_edges)
    _, V_abs, _ = _element_contributions_from_edges(edges_def)

    delta = _source_delta(G_flat, use_abs_J, alpha_nl)
    delta_per_tv = delta[conn] * (V_abs[:, None] / 4.0)
    f = jax.ops.segment_sum(delta_per_tv.reshape(-1), conn.reshape(-1),
                            num_segments=N**3)
    return f - f.mean()


def _correction(G, boxsize: float, N: int, cg_tol: float, cg_maxiter: int,
                use_abs_J: bool, alpha_nl: float) -> Array:
    K = assemble_stiffness_stencil(G, boxsize)
    f = _assemble_source(G, boxsize, use_abs_J, alpha_nl)
    phi = solve_poisson(K, f, tol=cg_tol, maxiter=cg_maxiter, boxsize=boxsize)
    return -fourier_hessian(phi.reshape(N, N, N), boxsize)


def _apply_drift(G: Array, dG_corr: Array, g_T: float, g_S: float,
                 scale: float) -> Array:
    tr = jnp.einsum("...ii->...", dG_corr)
    iso = (tr / 3.0)[..., None, None] * jnp.eye(3)[None, None, None, :, :]
    shear = dG_corr - iso
    return G + scale * (g_T * iso + g_S * shear)


# ---------------------------------------------------------------- increments


def _inv_laplacian_hessian(src: Array, boxsize: float, N: int) -> Array:
    """Return ``-∂_i∂_j ∇⁻²(src)`` as a ``(N, N, N, 3, 3)`` field.

    Used to convert a real-space scalar quantity into the corresponding
    ψ-gradient-tensor ("G"-style) correction. Reuses the existing
    :func:`fourier_hessian` machinery.
    """
    kd = get_fourier_grid([N, N, N], boxsize=boxsize, sparse_k_vecs=True,
                          full=False, dtype_num=32, relative=False,
                          with_jax=True)
    k_vecs = kd["k_vecs"]
    ksq = sum(k**2 for k in k_vecs)
    ksq_safe = jnp.where(ksq > 0, ksq, 1.0)

    src_hat = jnp.fft.rfftn(src)
    phi_hat = jnp.where(ksq > 0, src_hat / ksq_safe, 0.0)
    phi = jnp.fft.irfftn(phi_hat, s=(N, N, N))
    return -fourier_hessian(phi, boxsize)


def _non_gaussian_increment(dG1: Array, boxsize: float, N: int,
                            beta_nl: float, gamma_nl: float) -> Array:
    """Build the 2LPT + 3LPT admixture to ``dG1``.

    ``β · ∇∇⁻²[δ₁² − G₁:G₁]`` is the 2LPT piece (Bouchet β=3/7).
    ``γ · ∇∇⁻²[δ₁³ − 3 δ₁·G₁:G₁]`` is the 3LPT scalar-longitudinal
    piece (Bernardeau γ=5/21 is the bare UV value). The subtraction
    inside each bracket is what renders the quantity rotationally
    invariant in the sense of LPT kernels.
    """
    delta1 = jnp.einsum("...ii->...", dG1)
    GG = jnp.einsum("...ij,...ji->...", dG1, dG1)

    src_2 = beta_nl * (delta1 * delta1 - GG)
    src_3 = gamma_nl * (delta1 ** 3 - 3.0 * delta1 * GG)
    src = src_2 + src_3
    return _inv_laplacian_hessian(src, boxsize, N)


# ---------------------------------------------------------------- step / run


@partial(jax.jit, static_argnames=("N", "cg_maxiter", "method", "use_abs_J"))
def _step_v4(G, key, dsigma2, g_T, g_S, alpha_nl, beta_nl, gamma_nl,
             boxsize, N, cg_tol, cg_maxiter, method, use_abs_J):
    dG1 = one_lpt_increment(key, N, boxsize, dsigma2)
    dG_nl = _non_gaussian_increment(dG1, boxsize, N, beta_nl, gamma_nl)
    G_noise = G + dG1 + dG_nl

    if method == "euler":
        F = _correction(G_noise, boxsize, N, cg_tol, cg_maxiter,
                        use_abs_J, alpha_nl)
        return _apply_drift(G_noise, F, g_T, g_S, dsigma2)

    F1 = _correction(G_noise, boxsize, N, cg_tol, cg_maxiter, use_abs_J, alpha_nl)
    G_pred = _apply_drift(G_noise, F1, g_T, g_S, dsigma2)
    F2 = _correction(G_pred, boxsize, N, cg_tol, cg_maxiter, use_abs_J, alpha_nl)
    F_avg = 0.5 * (F1 + F2)
    return _apply_drift(G_noise, F_avg, g_T, g_S, dsigma2)


def fem_cascade_v4(
    sigma2: float,
    n_steps: int,
    N: int,
    boxsize: float,
    key: Array,
    g_T: float = 0.0,
    g_S: float | None = None,
    alpha_nl: float = 0.0,
    beta_nl: float = 0.0,
    gamma_nl: float = 0.0,
    method: Literal["euler", "midpoint"] = "midpoint",
    use_abs_J: bool = True,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
) -> CascadeV4Result:
    """Cascade v4 with CLPT-style cubic increment.

    :param beta_nl: 2LPT-style quadratic coupling (v3-compatible).
        ``3/7 ≈ 0.43`` is Bouchet's bare UV value; the RG-flowed IR
        value (empirically fit against PM) is negative.
    :param gamma_nl: 3LPT-style cubic coupling. ``5/21 ≈ 0.238`` is
        Bernardeau's bare value; the IR-flowed value should also be
        scanned against PM.
    """
    if g_S is None:
        g_S = g_T
    dsigma2 = sigma2 / n_steps
    subkeys = jax.random.split(key, n_steps)

    G = jnp.zeros((N, N, N, 3, 3))
    for step in range(n_steps):
        G = _step_v4(G, subkeys[step], dsigma2, g_T, g_S,
                     alpha_nl, beta_nl, gamma_nl,
                     boxsize, N, cg_tol, cg_maxiter, method, use_abs_J)

    F = jnp.eye(3) + G
    J = jnp.linalg.det(F)
    delta = 1.0 / J - 1.0
    sigma2_ach = float(jnp.var(jnp.einsum("...ii->...", G)))
    return CascadeV4Result(J=J, G=G, delta=delta, sigma2_trG=sigma2_ach)
