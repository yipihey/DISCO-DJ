"""Cascade v3: non-Gaussian physics additions on top of v2.

Motivating observation (from the v2→PM fits): with only a Gaussian 1LPT
increment + linear FEM-tidal correction, σ²_J tracks PM well but ``S₃(J)``
is systematically **too high** (~0.8 vs PM's ~0.4 at σ²=0.15). The tidal
correction of v2 is linear in ``δ`` — it damps variance but cannot alter
skewness. To match PM's J-PDF shape the cascade needs a genuinely
non-Gaussian ingredient.

Two independently-tuned knobs, both switchable via scalar coupling:

1. **Nonlinear Poisson source** (``alpha_nl``):
   Use ``δ_eff = δ + α · δ²`` inside the Poisson solve, turning the
   correction into a physically motivated 2LPT-kernel on the nodal
   overdensity. With ``α = 17/21`` this matches the 2LPT nonlinear
   correction at the single-mode level (Bernardeau 1994). At ``α = 0``
   this reduces to v2.

2. **Nonlinear increment coupling** (``beta_nl``):
   After drawing the 1LPT Gaussian increment ``dG₁``, add a 2LPT-style
   quadratic mixing ``β · 2LPT(dG₁)`` implemented via the standard
   Zel'dovich / 2LPT kernel (F₂-like). Non-Gaussianises the *noise*
   directly.  ``β = 3/7`` is the 2LPT Bouchet coefficient.

Both default to zero (reduces exactly to v2). Running with them tuned is
where the cascade starts to be physics-complete.
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


__all__ = ["fem_cascade_v3", "fem_cascade_v3_batch", "CascadeV3Result"]


@dataclass
class CascadeV3Result:
    J: Array
    G: Array
    delta: Array
    sigma2_trG: Array   # scalar Array for single call, (B,) for batched


def _source_delta(G_flat: Array, use_abs_J: bool, alpha_nl: float) -> Array:
    J = jnp.linalg.det(jnp.eye(3) + G_flat)
    J_eff = jnp.abs(J) if use_abs_J else J
    delta = 1.0 / J_eff - 1.0
    return delta + alpha_nl * delta * delta


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


def _non_gaussian_increment(dG1: Array, boxsize: float, N: int, beta_nl,
                            cg_tol: float, cg_maxiter: int) -> Array:
    """2LPT-style non-Gaussian admixture to a Gaussian 1LPT increment.

    The 2LPT displacement is the gradient of the solution of
    ``∇²φ₂ = (3/7)[(∇²ψ₁)² − (∂_i∂_j ψ₁)²]``. Since ``∇²ψ₁ = tr(G₁) = δ₁``
    and ``(∂_i∂_j ψ₁) = G₁_{ij}``, the RHS is
    ``(3/7)[δ₁² − Tr(G₁·G₁)]`` — i.e. the non-Gaussian piece that gives
    rise to the genuine 2LPT skewness. Multiplying by the inverse
    Laplacian, then taking the Hessian, builds the 2LPT correction to G.
    """
    delta1 = jnp.einsum("...ii->...", dG1)
    GG = jnp.einsum("...ij,...ji->...", dG1, dG1)
    src = beta_nl * (delta1 * delta1 - GG)

    kd = get_fourier_grid([N, N, N], boxsize=boxsize, sparse_k_vecs=True,
                          full=False, dtype_num=32, relative=False,
                          with_jax=True)
    k_vecs = kd["k_vecs"]
    ksq = sum(k**2 for k in k_vecs)
    ksq_safe = jnp.where(ksq > 0, ksq, 1.0)

    src_hat = jnp.fft.rfftn(src)
    phi2_hat = jnp.where(ksq > 0, src_hat / ksq_safe, 0.0)
    phi2 = jnp.fft.irfftn(phi2_hat, s=(N, N, N))
    return -fourier_hessian(phi2, boxsize)


def _apply_drift(G: Array, dG_corr: Array, g_T: float, g_S: float,
                 scale: float) -> Array:
    tr = jnp.einsum("...ii->...", dG_corr)
    iso = (tr / 3.0)[..., None, None] * jnp.eye(3)[None, None, None, :, :]
    shear = dG_corr - iso
    return G + scale * (g_T * iso + g_S * shear)


# ---------------------------------------------------------------------------
# Single-step bodies (traceable; called inside scan or as stand-alone jits).
# ---------------------------------------------------------------------------

def _step_v3_body_full(G, key, dsigma2, g_T, g_S, alpha_nl, beta_nl, k_cut,
                       pk_k, pk_Pk, boxsize, N,
                       cg_tol, cg_maxiter, method, use_abs_J):
    dG1 = one_lpt_increment(key, N, boxsize, dsigma2,
                            k_cut=k_cut, pk_k=pk_k, pk_Pk=pk_Pk)
    dG2 = _non_gaussian_increment(dG1, boxsize, N, beta_nl,
                                  cg_tol, cg_maxiter)
    G_noise = G + dG1 + dG2

    if method == "euler":
        dG_corr = _correction(G_noise, boxsize, N, cg_tol, cg_maxiter,
                              use_abs_J, alpha_nl)
        return _apply_drift(G_noise, dG_corr, g_T, g_S, dsigma2)

    F1 = _correction(G_noise, boxsize, N, cg_tol, cg_maxiter, use_abs_J, alpha_nl)
    G_pred = _apply_drift(G_noise, F1, g_T, g_S, dsigma2)
    F2 = _correction(G_pred, boxsize, N, cg_tol, cg_maxiter, use_abs_J, alpha_nl)
    F_avg = 0.5 * (F1 + F2)
    return _apply_drift(G_noise, F_avg, g_T, g_S, dsigma2)


def _step_v3_body_free(G, key, dsigma2, beta_nl, k_cut,
                       pk_k, pk_Pk, boxsize, N):
    """Drift-free step body. Skip the FEM Poisson solve because at
    ``g_T = g_S = 0`` the correction result is multiplied by zero in
    :func:`_apply_drift`. Saves ~3-5× per step (two CG solves in midpoint).
    """
    dG1 = one_lpt_increment(key, N, boxsize, dsigma2,
                            k_cut=k_cut, pk_k=pk_k, pk_Pk=pk_Pk)
    dG2 = _non_gaussian_increment(dG1, boxsize, N, beta_nl, 1e-7, 10)
    return G + dG1 + dG2


# Backward-compat single-step jit (kept — some tests call it directly).
@partial(jax.jit, static_argnames=("N", "cg_maxiter", "method", "use_abs_J"))
def _step_v3(G, key, dsigma2, g_T, g_S, alpha_nl, beta_nl, k_cut,
             pk_k, pk_Pk, boxsize, N,
             cg_tol, cg_maxiter, method, use_abs_J):
    return _step_v3_body_full(
        G, key, dsigma2, g_T, g_S, alpha_nl, beta_nl, k_cut,
        pk_k, pk_Pk, boxsize, N,
        cg_tol, cg_maxiter, method, use_abs_J)


# ---------------------------------------------------------------------------
# Full-trajectory jitted helpers (n_steps loop via jax.lax.scan).
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_steps", "N"))
def _run_cascade_v3_free(sigma2, beta_nl, key, n_steps, N, boxsize,
                         k_cut, pk_k, pk_Pk):
    dsigma2 = sigma2 / n_steps
    subkeys = jax.random.split(key, n_steps)

    def step_fn(G, k):
        return _step_v3_body_free(G, k, dsigma2, beta_nl, k_cut,
                                  pk_k, pk_Pk, boxsize, N), None

    G_init = jnp.zeros((N, N, N, 3, 3), dtype=jnp.float32)
    G, _ = jax.lax.scan(step_fn, G_init, subkeys)
    return G


@partial(jax.jit, static_argnames=("n_steps", "N", "cg_maxiter",
                                   "method", "use_abs_J"))
def _run_cascade_v3_full(sigma2, beta_nl, key, n_steps, N, boxsize,
                         g_T, g_S, alpha_nl, k_cut, pk_k, pk_Pk,
                         cg_tol, cg_maxiter, method, use_abs_J):
    dsigma2 = sigma2 / n_steps
    subkeys = jax.random.split(key, n_steps)

    def step_fn(G, k):
        return _step_v3_body_full(
            G, k, dsigma2, g_T, g_S, alpha_nl, beta_nl, k_cut,
            pk_k, pk_Pk, boxsize, N,
            cg_tol, cg_maxiter, method, use_abs_J), None

    G_init = jnp.zeros((N, N, N, 3, 3), dtype=jnp.float32)
    G, _ = jax.lax.scan(step_fn, G_init, subkeys)
    return G


def _finalise(G: Array) -> tuple[Array, Array, Array]:
    F = jnp.eye(3) + G
    J = jnp.linalg.det(F)
    delta = 1.0 / J - 1.0
    sigma2_trG = jnp.var(jnp.einsum("...ii->...", G))
    return J, delta, sigma2_trG


def _drift_is_zero(g_T: float, g_S: float | None) -> bool:
    return g_T == 0.0 and (g_S is None or g_S == 0.0)


# ---------------------------------------------------------------------------
# Public API: single-call and batch.
# ---------------------------------------------------------------------------

def fem_cascade_v3(
    sigma2: float,
    n_steps: int,
    N: int,
    boxsize: float,
    key: Array,
    g_T: float = 0.0,
    g_S: float | None = None,
    alpha_nl: float = 0.0,
    beta_nl: float = 0.0,
    k_cut: float | None = None,
    pk_k: Array | None = None,
    pk_Pk: Array | None = None,
    method: Literal["euler", "midpoint"] = "midpoint",
    use_abs_J: bool = True,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
) -> CascadeV3Result:
    """Cascade v3 with non-Gaussian physics extensions.

    :param alpha_nl: coefficient of ``δ²`` in the nonlinear Poisson source.
        ``17/21 ≈ 0.81`` matches 2LPT at a single mode (Bernardeau 1994).
    :param beta_nl: coefficient of the 2LPT non-Gaussian increment.
        ``3/7 ≈ 0.43`` is Bouchet's 2LPT coupling.

    When ``g_T = g_S = 0`` (default), the expensive FEM Poisson correction
    is skipped — it would be multiplied by zero in the drift update.
    """
    if _drift_is_zero(g_T, g_S):
        G = _run_cascade_v3_free(
            jnp.asarray(sigma2, dtype=jnp.float32),
            jnp.asarray(beta_nl, dtype=jnp.float32),
            key, n_steps, N, boxsize, k_cut, pk_k, pk_Pk)
    else:
        g_S_val = g_T if g_S is None else g_S
        G = _run_cascade_v3_full(
            jnp.asarray(sigma2, dtype=jnp.float32),
            jnp.asarray(beta_nl, dtype=jnp.float32),
            key, n_steps, N, boxsize,
            g_T, g_S_val, alpha_nl, k_cut, pk_k, pk_Pk,
            cg_tol, cg_maxiter, method, use_abs_J)

    J, delta, sigma2_trG = _finalise(G)
    return CascadeV3Result(J=J, G=G, delta=delta, sigma2_trG=sigma2_trG)


def fem_cascade_v3_batch(
    sigma2,
    beta_nl,
    keys: Array,
    n_steps: int,
    N: int,
    boxsize: float,
    g_T: float = 0.0,
    g_S: float | None = None,
    alpha_nl: float = 0.0,
    k_cut: float | None = None,
    pk_k: Array | None = None,
    pk_Pk: Array | None = None,
    method: Literal["euler", "midpoint"] = "midpoint",
    use_abs_J: bool = True,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
    chunk_size: int | None = None,
) -> CascadeV3Result:
    """Batched cascade over ``(sigma2, beta_nl, keys)`` via ``vmap``.

    ``sigma2`` may be a scalar (broadcast) or shape ``(B,)``. ``beta_nl``
    shape ``(B,)``. ``keys`` shape ``(B, 2)``. All per-snapshot inputs
    (``pk_k``, ``pk_Pk``, ``k_cut``, cosmology) are shared across the
    batch.

    ``chunk_size`` splits the batch into Python-looped chunks to bound
    peak memory (useful at ``N ≥ 128``). If ``None`` the whole batch
    runs in a single compiled call.
    """
    beta_nl = jnp.asarray(beta_nl, dtype=jnp.float32)
    keys = jnp.asarray(keys)
    B = int(beta_nl.shape[0])

    sigma2_arr = jnp.asarray(sigma2, dtype=jnp.float32)
    if sigma2_arr.ndim == 0:
        sigma2_arr = jnp.broadcast_to(sigma2_arr, (B,))

    if _drift_is_zero(g_T, g_S):
        def core(s2, b, k):
            return _run_cascade_v3_free(
                s2, b, k, n_steps, N, boxsize, k_cut, pk_k, pk_Pk)
    else:
        g_S_val = g_T if g_S is None else g_S
        def core(s2, b, k):
            return _run_cascade_v3_full(
                s2, b, k, n_steps, N, boxsize,
                g_T, g_S_val, alpha_nl, k_cut, pk_k, pk_Pk,
                cg_tol, cg_maxiter, method, use_abs_J)

    batched = jax.vmap(core, in_axes=(0, 0, 0))

    if chunk_size is None or chunk_size >= B:
        G = batched(sigma2_arr, beta_nl, keys)
    else:
        parts = []
        for start in range(0, B, chunk_size):
            stop = min(start + chunk_size, B)
            parts.append(batched(sigma2_arr[start:stop],
                                 beta_nl[start:stop],
                                 keys[start:stop]))
        G = jnp.concatenate(parts, axis=0)

    F = jnp.eye(3) + G
    J = jnp.linalg.det(F)
    delta = 1.0 / J - 1.0
    # Per-batch trace variance: var over spatial dims for each batch element.
    trG = jnp.einsum("b...ii->b...", G)          # shape (B, N, N, N)
    sigma2_trG = jnp.var(trG.reshape(B, -1), axis=1)
    return CascadeV3Result(J=J, G=G, delta=delta, sigma2_trG=sigma2_trG)
