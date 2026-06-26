"""FEM cascade: iterative free Gaussian + Poisson-correction evolution of G.

See :mod:`discodj.fem` for the underlying FEM machinery. The cascade evolves
the deformation gradient ``G = ∂ψ/∂q`` from zero to a target variance ``σ²``
in ``n_steps`` stages. Each stage:

1. Adds an independent 1LPT increment ``dG`` with ``Var[tr(dG)] = σ²/n_steps``.
2. Assembles the FEM stiffness on the current deformed mesh and solves the
   Poisson equation for the gravitational potential ``φ``.
3. Applies the correction ``G += g2c · δG · dσ²`` where
   ``δG_ij = −∂_i ∂_j φ`` (computed via the Fourier Hessian).

Sign convention: in overdense regions (``φ < 0``), ``δG`` has negative trace,
enhancing collapse. This should drive S₃ down from the free Zel'dovich value
(~2.0) toward the RG-flow prediction (~1.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ..core.grids import get_fourier_grid
from ..core.kernels import gradient_kernel, inv_laplace_kernel
from ..fem.assembly import assemble_stiffness_stencil, assemble_source
from ..fem.solve import solve_poisson
from ..fem.hessian import fourier_hessian


__all__ = ["fem_cascade", "one_lpt_increment", "pk_table_from_discodj",
           "CascadeResult"]


def pk_table_from_discodj(dj) -> tuple[Array, Array]:
    """Return ``(pk_k, pk_Pk)`` arrays from a configured DiscoDJ object.

    The arrays can be fed directly to :func:`one_lpt_increment` and the
    v3 cascade's ``pk_k, pk_Pk`` arguments. Requires that the caller has
    already called ``with_timetables`` and ``with_linear_ps`` on ``dj``.
    """
    k = jnp.asarray(dj._pk_table["k"])
    Pk = jnp.asarray(dj._pk_table["Pk"])
    return k, Pk


@dataclass
class CascadeResult:
    """Output of :func:`fem_cascade`.

    Access the fields that matter for diagnostics: ``J`` for the volume
    statistics (mean, variance, skewness, ξ̄), and ``G`` if the caller wants
    to compute eigenvalues or feed downstream analyses.
    """

    J: Array        # (N, N, N) det(I + G) at each vertex
    G: Array        # (N, N, N, 3, 3) final deformation gradient
    delta: Array    # (N, N, N) 1/J − 1 at each vertex
    sigma2: float   # Achieved trace variance (diagnostic)


def _k_dict(N: int, boxsize: float):
    return get_fourier_grid(
        [N, N, N], boxsize=boxsize, sparse_k_vecs=True, full=False,
        dtype_num=32, relative=False, with_jax=True,
    )


def one_lpt_increment(key: Array, N: int, boxsize: float, dsigma2: float,
                      k_cut: float | None = None,
                      pk_k: Array | None = None,
                      pk_Pk: Array | None = None) -> Array:
    """Draw a fresh 1LPT deformation-gradient increment.

    Generates Gaussian white-noise ``z``, takes the FFT, optionally shapes
    its power spectrum with either:

    - ``k_cut``: Gaussian cutoff ``exp(-½(k/k_cut)²)`` — simple IR filter,
      matches PM sims initialised with the same cutoff.
    - ``(pk_k, pk_Pk)``: a full tabulated linear P(k) (e.g. LCDM). Window
      is ``sqrt(P_interp(|k|))``. See :func:`pk_table_from_discodj`.

    After windowing we renormalise so ``Var[tr(dG)] = dσ²``. Then project
    onto the 1LPT tensor ``G_ij(k) = (k_i k_j / |k|²) ẑ(k)``. Without
    windowing, ``z`` is white noise; on a cubic grid the resulting G has
    ``σ²_G11/σ²_G12 ≈ 2.36`` (intrinsic cubic-lattice value) vs PM's ~3.0
    with LCDM ICs. Shaping ``z`` with the target P(k) restores the right
    tensor structure.

    :param key: JAX PRNGKey.
    :param N: grid resolution.
    :param boxsize: box size (Mpc/h).
    :param dsigma2: target ``Var[tr(dG)]`` after windowing.
    :param k_cut: Gaussian cutoff in h/Mpc.
    :param pk_k, pk_Pk: arrays tabulating a P(k). Takes precedence over
        ``k_cut`` if both are provided.
    """
    z = jax.random.normal(key, shape=(N, N, N))
    z_hat = jnp.fft.rfftn(z)

    kd = _k_dict(N, boxsize)
    k_vecs = kd["k_vecs"]
    kmag = kd["|k|"]

    if pk_k is not None and pk_Pk is not None:
        P_on_grid = jnp.interp(kmag, pk_k, pk_Pk,
                               left=pk_Pk[0], right=pk_Pk[-1])
        z_hat = z_hat * jnp.sqrt(jnp.maximum(P_on_grid, 0.0))
    elif k_cut is not None:
        z_hat = z_hat * jnp.exp(-0.5 * (kmag / k_cut) ** 2)
    # Renormalise to the requested variance of tr(dG) = z.
    z_filtered = jnp.fft.irfftn(z_hat, s=(N, N, N))
    var = jnp.var(z_filtered)
    scale = jnp.where(var > 0, jnp.sqrt(dsigma2 / jnp.maximum(var, 1e-30)),
                      0.0)
    z_hat = z_hat * scale

    # Use DISCO-DJ's ``gradient_kernel`` (``i·k`` with Nyquist zeroed) and
    # ``inv_laplace_kernel`` (``-1/|k|²``) to build the 1LPT projection
    # consistently with how DISCO-DJ's PM generates its G. This removes a
    # subtle lattice-anisotropy bias that was present when we built
    # ``k_i k_j / |k|²`` directly from raw ``k_vecs``: the Nyquist modes
    # in each axis were *not* being zeroed, which shifted the per-component
    # variance ratio ``⟨G_ii²⟩ / ⟨G_ij²⟩`` from the ideal 3 toward ~2.4 at
    # N=32. With the gradient/inv-Laplace split the lattice ratio matches
    # DISCO-DJ's own 1LPT to within sampling noise.
    inv_lap = inv_laplace_kernel(k_vecs, order=0, with_jax=True)
    grads = [gradient_kernel(k_vecs, d, order=0, with_jax=True)
             for d in range(3)]

    # ψ_1 in Fourier: (i k_d) · (-1/|k|²) · z_hat.  Then G_ij = ∂_i ψ_j.
    # G is symmetric (G_ij = -k_i k_j / |k|² · z_hat), so compute the 6
    # unique components and symmetrise on assignment.
    dG = jnp.zeros((N, N, N, 3, 3), dtype=jnp.float32)
    for i in range(3):
        for j in range(i, 3):
            proj = grads[i] * grads[j] * inv_lap
            comp = jnp.fft.irfftn(proj * z_hat, s=(N, N, N))
            dG = dG.at[..., i, j].set(comp)
            if i != j:
                dG = dG.at[..., j, i].set(comp)
    return dG


@partial(jax.jit, static_argnames=("N", "cg_maxiter"))
def _cascade_step(G, key, dsigma2, g2c, boxsize, N, cg_tol, cg_maxiter):
    """One cascade step, fused into a single jitted function.

    Draws a fresh 1LPT increment, adds it to ``G``, solves the FEM Poisson
    problem on the deformed mesh, and applies the tidal correction.
    """
    dG = one_lpt_increment(key, N, boxsize, dsigma2)
    G1 = G + dG
    K = assemble_stiffness_stencil(G1, boxsize)
    f = assemble_source(G1, boxsize)
    phi = solve_poisson(K, f, tol=cg_tol, maxiter=cg_maxiter, boxsize=boxsize)
    dG_corr = -fourier_hessian(phi.reshape(N, N, N), boxsize)
    return G1 + g2c * dG_corr * dsigma2


def fem_cascade(
    sigma2: float,
    n_steps: int,
    N: int,
    boxsize: float,
    key: Array,
    g2c: float = 0.05,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
) -> CascadeResult:
    """Run the FEM-corrected Zel'dovich cascade.

    :param sigma2: target variance of ``tr(G)`` at the final step.
    :param n_steps: number of cascade stages (each adds ``sigma2/n_steps``).
    :param N: grid resolution per dimension.
    :param boxsize: simulation box size.
    :param key: JAX PRNGKey. Split internally into ``n_steps`` subkeys.
    :param g2c: amplitude of the tidal correction. Physical value to be
        tuned against nLPT; task spec suggests starting at 0.05 and scanning.
    :param cg_tol: tolerance for the Poisson CG solver.
    :param cg_maxiter: iteration cap for the Poisson CG solver.
    :return: :class:`CascadeResult` with final ``G``, ``J``, and ``δ = 1/J−1``.
    """
    dsigma2 = sigma2 / n_steps
    subkeys = jax.random.split(key, n_steps)

    G = jnp.zeros((N, N, N, 3, 3))

    for step in range(n_steps):
        G = _cascade_step(
            G, subkeys[step], dsigma2, g2c, boxsize, N, cg_tol, cg_maxiter,
        )

    F = jnp.eye(3) + G
    J = jnp.linalg.det(F)
    delta = 1.0 / J - 1.0

    # Report the actually-realised trace variance (diagnostic).
    sigma2_achieved = float(jnp.var(jnp.einsum("...ii->...", G)))

    return CascadeResult(J=J, G=G, delta=delta, sigma2=sigma2_achieved)
