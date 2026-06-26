"""Cascade v5: fix the NLPT increment scaling.

In v3/v4 the 2LPT/3LPT admixture was built from the *per-step* Gaussian
increment ``dG₁``, so each step added a correction of variance
``O(β² · dσ²²)``.  Summing ``n_steps`` of these gives
``Var ~ β² · σ² · dσ²``, which *vanishes* in the continuum
``n_steps → ∞`` limit — the v3/v4 non-Gaussian contribution was really a
finite-step artifact.

Physically, the 2LPT and 3LPT corrections are defined at the
*cumulative* linear field. So per-step the differential correction
should be the CROSS TERM between the accumulated Gaussian and the new
increment:

    d[∇∇⁻²(δ₁² − G₁:G₁)] = 2·δ_{acc}·dδ_1 − 2·G_{acc}:dG_1 + O(dσ⁴)

which scales as ``σ · dσ`` per step — small each step, accumulating
correctly to the full 2LPT at final σ². The same applies to 3LPT
(scales as ``σ² · dσ``).

Cleanest implementation: maintain the cumulative Gaussian field
``G_linear`` as cascade state and evaluate the full 2LPT/3LPT
at the end. Per-step this is equivalent to the running differential
formulation; the difference appears only when the FEM tidal correction
is actively nonlinear.

This ``v5`` is the resummed cascade with ``β, γ`` still tunable against
PM. For ``g_T = 0`` it is literally "apply 2LPT and 3LPT to a
Gaussian field of variance σ²" — the canonical Eulerian perturbation
theory answer. For ``g_T ≠ 0`` the cascade adds on top of that a
FEM-derived tidal feedback, per-step.
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


__all__ = ["fem_cascade_v5", "CascadeV5Result"]


@dataclass
class CascadeV5Result:
    J: Array
    G: Array
    G_linear: Array
    delta: Array
    sigma2_trG: Array


def _nlpt_nongaussian(G_linear: Array, boxsize: float, N: int,
                      beta_nl: float, gamma_nl: float) -> Array:
    """Evaluate 2LPT + 3LPT corrections at the cumulative Gaussian field.

    ``G_linear`` is the accumulated 1LPT deformation gradient. Returns
    the non-Gaussian piece ``β·G_2LPT + γ·G_3LPT`` with the same (N,N,N,3,3)
    shape.
    """
    delta1 = jnp.einsum("...ii->...", G_linear)
    GG = jnp.einsum("...ij,...ji->...", G_linear, G_linear)

    src_2 = beta_nl * (delta1 * delta1 - GG)
    src_3 = gamma_nl * (delta1 ** 3 - 3.0 * delta1 * GG)
    src = src_2 + src_3

    kd = get_fourier_grid([N, N, N], boxsize=boxsize, sparse_k_vecs=True,
                          full=False, dtype_num=32, relative=False,
                          with_jax=True)
    k_vecs = kd["k_vecs"]
    ksq = sum(k**2 for k in k_vecs)
    ksq_safe = jnp.where(ksq > 0, ksq, 1.0)

    phi_hat = jnp.where(ksq > 0, jnp.fft.rfftn(src) / ksq_safe, 0.0)
    phi = jnp.fft.irfftn(phi_hat, s=(N, N, N))
    return -fourier_hessian(phi, boxsize)


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
    d = _source_delta(G_flat, use_abs_J, alpha_nl)
    d_tv = d[conn] * (V_abs[:, None] / 4.0)
    f = jax.ops.segment_sum(d_tv.reshape(-1), conn.reshape(-1),
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


@partial(jax.jit,
         static_argnames=("N", "cg_maxiter", "method", "use_abs_J"))
def _step_v5(G_linear, G_tidal, key, dsigma2, g_T, g_S,
             alpha_nl, beta_nl, gamma_nl,
             boxsize, N, cg_tol, cg_maxiter, method, use_abs_J):
    """Advance ``(G_linear, G_tidal)`` by one cascade step.

    The full G used for the FEM correction is
    ``G_total = G_linear_new + NLPT(G_linear_new) + G_tidal``.
    """
    dG1 = one_lpt_increment(key, N, boxsize, dsigma2)
    G_linear_new = G_linear + dG1
    G_nl = _nlpt_nongaussian(G_linear_new, boxsize, N, beta_nl, gamma_nl)

    # Always run the FEM correction machinery; (g_T, g_S) = (0, 0) makes
    # it a no-op naturally. Python-level branching on traced values breaks
    # jit (we hit this on v3 earlier).
    G_for_fem = G_linear_new + G_nl + G_tidal
    if method == "euler":
        F = _correction(G_for_fem, boxsize, N, cg_tol, cg_maxiter,
                        use_abs_J, alpha_nl)
        return G_linear_new, _apply_drift(G_tidal, F, g_T, g_S, dsigma2)

    F1 = _correction(G_for_fem, boxsize, N, cg_tol, cg_maxiter, use_abs_J, alpha_nl)
    G_pred = G_linear_new + G_nl + _apply_drift(G_tidal, F1, g_T, g_S, dsigma2)
    F2 = _correction(G_pred, boxsize, N, cg_tol, cg_maxiter, use_abs_J, alpha_nl)
    F_avg = 0.5 * (F1 + F2)
    return G_linear_new, _apply_drift(G_tidal, F_avg, g_T, g_S, dsigma2)


@partial(jax.jit, static_argnames=("n_steps", "N",
                                   "k_cut_is_none", "pk_is_none"))
def _run_cascade_v5_free(sigma2, beta_nl, gamma_nl, key,
                          n_steps, N, boxsize, k_cut, pk_k, pk_Pk,
                          k_cut_is_none, pk_is_none):
    """Drift-free cascade v5 (``g_T = g_S = 0``).

    Sum 1LPT increments over n_steps into a cumulative ``G_linear``, then
    apply ``β·G_2LPT + γ·G_3LPT`` once at the end. Avoids the expensive
    FEM Poisson solve (which would contribute zero anyway). The ``_is_none``
    flags are workaround statics because the jit signature can't hold
    ``None`` as a traced argument.
    """
    dsigma2 = sigma2 / n_steps
    subkeys = jax.random.split(key, n_steps)

    def step_fn(G_linear, k):
        dG1 = one_lpt_increment(
            k, N, boxsize, dsigma2,
            k_cut=None if k_cut_is_none else k_cut,
            pk_k=None if pk_is_none else pk_k,
            pk_Pk=None if pk_is_none else pk_Pk)
        return G_linear + dG1, None

    G_init = jnp.zeros((N, N, N, 3, 3), dtype=jnp.float32)
    G_linear, _ = jax.lax.scan(step_fn, G_init, subkeys)
    G_nl = _nlpt_nongaussian(G_linear, boxsize, N, beta_nl, gamma_nl)
    return G_linear, G_linear + G_nl


def fem_cascade_v5(
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
    k_cut: float | None = None,
    pk_k: Array | None = None,
    pk_Pk: Array | None = None,
    method: Literal["euler", "midpoint"] = "midpoint",
    use_abs_J: bool = True,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
) -> CascadeV5Result:
    """Cascade with cumulative-Gaussian-based NLPT injection.

    The 2LPT and 3LPT corrections are evaluated at the *accumulated*
    Gaussian field rather than at each step's increment — this is the
    correct scaling for resumming nLPT through a cascade (cf. v3/v4
    which had the 2LPT term scale as O(dσ⁴) per step, vanishing in
    the continuum limit).

    For ``g_T = g_S = 0`` the FEM tidal correction is skipped entirely
    and the whole trajectory runs in one jitted ``scan``.
    """
    if g_S is None:
        g_S = g_T

    drift_is_zero = (g_T == 0.0 and g_S == 0.0)
    k_cut_is_none = k_cut is None
    pk_is_none = pk_k is None or pk_Pk is None

    if drift_is_zero:
        # Dummy arrays for the jit signature when the real ones are None.
        k_cut_arg = 0.0 if k_cut_is_none else float(k_cut)
        pk_k_arg = jnp.zeros(1) if pk_is_none else pk_k
        pk_Pk_arg = jnp.zeros(1) if pk_is_none else pk_Pk
        G_linear, G_total = _run_cascade_v5_free(
            jnp.asarray(sigma2, dtype=jnp.float32),
            jnp.asarray(beta_nl, dtype=jnp.float32),
            jnp.asarray(gamma_nl, dtype=jnp.float32),
            key, n_steps, N, boxsize,
            k_cut_arg, pk_k_arg, pk_Pk_arg,
            k_cut_is_none, pk_is_none)
    else:
        dsigma2 = sigma2 / n_steps
        subkeys = jax.random.split(key, n_steps)

        G_linear = jnp.zeros((N, N, N, 3, 3))
        G_tidal = jnp.zeros((N, N, N, 3, 3))

        for step in range(n_steps):
            G_linear, G_tidal = _step_v5(
                G_linear, G_tidal, subkeys[step], dsigma2, g_T, g_S,
                alpha_nl, beta_nl, gamma_nl, boxsize, N,
                cg_tol, cg_maxiter, method, use_abs_J,
            )

        G_nl = _nlpt_nongaussian(G_linear, boxsize, N, beta_nl, gamma_nl)
        G_total = G_linear + G_nl + G_tidal

    J = jnp.linalg.det(jnp.eye(3) + G_total)
    delta = 1.0 / J - 1.0
    s2 = jnp.var(jnp.einsum("...ii->...", G_total))
    return CascadeV5Result(J=J, G=G_total, G_linear=G_linear,
                           delta=delta, sigma2_trG=s2)
