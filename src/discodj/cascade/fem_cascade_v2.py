"""Improved FEM cascade: midpoint (Heun) integrator + physics-first extensions.

Replaces the Euler splitting in :mod:`fem_cascade` with a trapezoidal /
Heun step on the drift term. Adds two knobs absent from v1:

- Separate trace/shear couplings ``g_T, g_S`` for the isotropic and
  traceless parts of the FEM tidal correction.
- Optional ``|J|`` in the Poisson source so the cascade stays physical
  after local shell crossings (``δ = 1/|J| - 1 ≥ -1``).

All cosmetic changes from v1 are preserved; importing both modules in the
same session is safe.
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
    FEMStencil, _element_contributions_from_edges, _edges_from_G,
    assemble_stiffness_stencil, stencil_matvec,
)
from ..fem.hessian import fourier_hessian
from ..fem.kuhn import kuhn_connectivity, lagrangian_edge_vectors
from ..fem.solve import solve_poisson
from .fem_cascade import one_lpt_increment


__all__ = ["fem_cascade_v2", "CascadeV2Result"]


@dataclass
class CascadeV2Result:
    J: Array              # (N, N, N) det(I + G)
    G: Array              # (N, N, N, 3, 3) final deformation gradient
    delta: Array          # 1/J − 1  (Eulerian overdensity, with sign)
    sigma2_trG: float     # Achieved Var[tr G]


def _source_delta(G_flat: Array, use_abs_J: bool) -> Array:
    """Vertex-wise Eulerian overdensity fed to the FEM Poisson source.

    :param use_abs_J: if True, ``δ = 1/|J| − 1`` (always ≥ −1, physical
        after tet inversion). If False, ``δ = 1/J − 1`` (v1 behaviour;
        blows up through J = 0).
    """
    J = jnp.linalg.det(jnp.eye(3) + G_flat)
    J_eff = jnp.abs(J) if use_abs_J else J
    return 1.0 / J_eff - 1.0


def _assemble_source_vertex(G, boxsize: float, use_abs_J: bool) -> Array:
    """Replacement for :func:`assemble_source` with the ``|J|`` option.

    Note: uses ``|V_tet|`` for the lumped mass, same as v1. The conceptual
    difference is only the ``J`` convention in the nodal ``δ``.
    """
    N = G.shape[0]
    G_flat = G.reshape(N**3, 3, 3)
    conn = kuhn_connectivity(N)
    lag_edges = lagrangian_edge_vectors(N, boxsize)
    edges_def = _edges_from_G(G_flat, conn, lag_edges)
    _, V_abs, _ = _element_contributions_from_edges(edges_def)

    delta = _source_delta(G_flat, use_abs_J)
    delta_per_tv = delta[conn] * (V_abs[:, None] / 4.0)
    f = jax.ops.segment_sum(delta_per_tv.reshape(-1), conn.reshape(-1),
                            num_segments=N**3)
    return f - f.mean()


def _correction(G, boxsize: float, N: int, cg_tol: float, cg_maxiter: int,
                use_abs_J: bool) -> Array:
    """Compute the raw FEM tidal correction ``δG_ij = −∂_i∂_j φ``.

    This is the "gravitational force direction" shared by all cascade
    variants; callers apply whatever coupling(s) they want on top.
    """
    K = assemble_stiffness_stencil(G, boxsize)
    f = _assemble_source_vertex(G, boxsize, use_abs_J)
    phi = solve_poisson(K, f, tol=cg_tol, maxiter=cg_maxiter, boxsize=boxsize)
    return -fourier_hessian(phi.reshape(N, N, N), boxsize)


def _apply_drift(G: Array, dG_corr: Array, g_T: float, g_S: float,
                 scale: float) -> Array:
    """Apply the ``(trace, shear)``-decomposed correction scaled by ``scale``.

    ``dG_corr`` is shape ``(N, N, N, 3, 3)``. The trace-isotropic piece is
    ``(1/3) · tr(dG_corr) · I``; the shear piece is the traceless remainder.
    For ``g_T == g_S`` this reduces to the single-coupling v1 form.
    """
    tr = jnp.einsum("...ii->...", dG_corr)
    iso = (tr / 3.0)[..., None, None] * jnp.eye(3)[None, None, None, :, :]
    shear = dG_corr - iso
    return G + scale * (g_T * iso + g_S * shear)


@partial(jax.jit,
         static_argnames=("N", "cg_maxiter", "method", "use_abs_J"))
def _cascade_step_v2(G, key, dsigma2, g_T, g_S, boxsize, N,
                     cg_tol, cg_maxiter, method, use_abs_J):
    """Single cascade step with the chosen integrator.

    ``method`` is ``"euler"`` (v1 behaviour) or ``"midpoint"`` (trapezoidal
    drift — 2nd-order strong for the deterministic part).
    """
    dG_free = one_lpt_increment(key, N, boxsize, dsigma2)
    G_noise = G + dG_free    # state after stochastic drift

    if method == "euler":
        dG_corr = _correction(G_noise, boxsize, N, cg_tol, cg_maxiter,
                              use_abs_J)
        return _apply_drift(G_noise, dG_corr, g_T, g_S, dsigma2)

    # Midpoint (Heun): predictor F(G_noise), corrector F(predictor).
    F1 = _correction(G_noise, boxsize, N, cg_tol, cg_maxiter, use_abs_J)
    G_pred = _apply_drift(G_noise, F1, g_T, g_S, dsigma2)
    F2 = _correction(G_pred, boxsize, N, cg_tol, cg_maxiter, use_abs_J)
    F_avg = 0.5 * (F1 + F2)
    return _apply_drift(G_noise, F_avg, g_T, g_S, dsigma2)


def fem_cascade_v2(
    sigma2: float,
    n_steps: int,
    N: int,
    boxsize: float,
    key: Array,
    g_T: float = 0.05,
    g_S: float | None = None,
    method: Literal["euler", "midpoint"] = "midpoint",
    use_abs_J: bool = True,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
) -> CascadeV2Result:
    """Run the improved cascade.

    :param g_T: trace-part coupling. Keeps the current cascade's
        "enhance collapse at overdensity" meaning.
    :param g_S: shear-part coupling. If ``None``, set to ``g_T`` (single
        coupling, v1-compatible physics).
    :param method: ``"midpoint"`` (default, 2nd-order drift) or
        ``"euler"`` (v1 behaviour for comparison).
    :param use_abs_J: if True (default), use ``|J|`` in the Eulerian δ
        source — physically correct through mild shell crossing; prevents
        the cascade from exploding at σ² ≳ 0.5.
    """
    if g_S is None:
        g_S = g_T
    dsigma2 = sigma2 / n_steps
    subkeys = jax.random.split(key, n_steps)

    G = jnp.zeros((N, N, N, 3, 3))
    for step in range(n_steps):
        G = _cascade_step_v2(
            G, subkeys[step], dsigma2, g_T, g_S, boxsize, N,
            cg_tol, cg_maxiter, method, use_abs_J,
        )

    F = jnp.eye(3) + G
    J = jnp.linalg.det(F)
    delta = 1.0 / J - 1.0  # reported with *signed* J so the caller sees
                            # post-shell-crossing tails honestly.

    sigma2_achieved = float(jnp.var(jnp.einsum("...ii->...", G)))
    return CascadeV2Result(J=J, G=G, delta=delta, sigma2_trG=sigma2_achieved)
