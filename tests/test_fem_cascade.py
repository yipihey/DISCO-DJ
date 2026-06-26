"""Tests for the FEM cascade and its components."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from discodj.fem.kuhn import (
    kuhn_connectivity,
    lagrangian_edge_vectors,
    _KUHN_TETS,
    _CORNER_OFFSETS,
)
from discodj.fem.assembly import assemble_stiffness, assemble_source
from discodj.fem.solve import solve_poisson
from discodj.fem.hessian import fourier_hessian
from discodj.cascade.fem_cascade import fem_cascade, one_lpt_increment


# ---------------------------------------------------------------- connectivity


@pytest.mark.parametrize("N", [2, 4, 8])
def test_kuhn_connectivity_shape_and_range(N):
    conn = kuhn_connectivity(N)
    assert conn.shape == (6 * N**3, 4)
    assert int(conn.min()) == 0
    assert int(conn.max()) == N**3 - 1


@pytest.mark.parametrize("N", [2, 4, 8])
def test_kuhn_tets_have_distinct_vertices(N):
    conn = np.asarray(kuhn_connectivity(N))
    sorted_c = np.sort(conn, axis=1)
    distinct = (sorted_c[:, :-1] != sorted_c[:, 1:]).all(axis=1)
    assert distinct.all(), "Each tet must have 4 distinct vertices"


@pytest.mark.parametrize("N", [2, 4, 8])
def test_undeformed_volumes_positive_and_tile_cube(N):
    edges = np.asarray(lagrangian_edge_vectors(N, boxsize=1.0))
    # Each of the 6 tets: volume = det(edges) / 6
    vols = np.linalg.det(edges) / 6.0
    assert (vols > 0).all(), f"Undeformed tet volumes must be positive: {vols}"
    h = 1.0 / N
    np.testing.assert_allclose(vols.sum(), h**3, atol=1e-12)


# ---------------------------------------------------------------- Laplacian


@pytest.mark.parametrize("precond", ["jacobi", "fft"])
def test_undeformed_poisson_converges_as_h_squared(precond):
    """Manufactured solution: cos(2π x/L). FEM error should be O((kh)²)."""
    # Jacobi needs many CG iterations for 3D Laplacian; the FFT precond
    # converges in 2–3 iters but drifts after ~10 in float32.
    maxiter, tol = (2000, 1e-10) if precond == "jacobi" else (3, 1e-10)

    errs = {}
    for N in (8, 16, 32):
        L = 1.0
        h = L / N
        G = jnp.zeros((N, N, N, 3, 3))
        K = assemble_stiffness(G, L)

        qs = jnp.arange(N) * h
        cos_kx = (jnp.cos(2 * jnp.pi * qs[:, None, None] / L)
                  * jnp.ones((N, N, N)))
        f_test = cos_kx.ravel() * h**3
        f_test = f_test - f_test.mean()
        phi = solve_poisson(K, f_test, tol=tol, maxiter=maxiter,
                            preconditioner=precond, boxsize=L)

        phi_ana = (L**2 / (4 * np.pi**2)) * np.asarray(cos_kx).ravel()
        phi_ana -= phi_ana.mean()
        errs[N] = float(np.linalg.norm(np.asarray(phi) - phi_ana)
                         / np.linalg.norm(phi_ana))

    # Second-order convergence: halving h should shrink error ~4x.
    assert errs[16] < 0.05
    assert errs[32] < 0.01
    assert errs[32] < errs[16] / 3.0


def test_undeformed_source_is_zero():
    N = 8
    G = jnp.zeros((N, N, N, 3, 3))
    f = assemble_source(G, boxsize=1.0)
    assert float(jnp.max(jnp.abs(f))) < 1e-6


# ---------------------------------------------------------------- Hessian


def test_fourier_hessian_analytic():
    """φ = cos(2π x/L) → ∂_i∂_j φ = -(2π/L)² cos(2π x/L) δ_{i,0} δ_{j,0}."""
    N = 16
    L = 1.0
    qs = jnp.arange(N) * (L / N)
    phi = jnp.cos(2 * jnp.pi * qs[:, None, None] / L) * jnp.ones((N, N, N))

    hess = fourier_hessian(phi, L)
    k = 2 * np.pi / L
    expected_00 = -(k**2) * np.asarray(phi)
    actual_00 = np.asarray(hess[..., 0, 0])
    np.testing.assert_allclose(actual_00, expected_00, atol=1e-3, rtol=1e-3)

    # Off-diagonal should be numerical zero.
    assert float(jnp.max(jnp.abs(hess[..., 0, 1]))) < 1e-4
    assert float(jnp.max(jnp.abs(hess[..., 1, 1]))) < 1e-4


# ---------------------------------------------------------------- cascade


def test_one_lpt_increment_trace_variance():
    key = jax.random.PRNGKey(0)
    dG = one_lpt_increment(key, N=16, boxsize=1.0, dsigma2=0.04)
    trace = jnp.einsum("...ii->...", dG)
    # Lattice-finite variance estimate has ~1/N^(3/2) fractional noise.
    assert 0.025 < float(jnp.var(trace)) < 0.055


def test_free_cascade_recovers_zeldovich_variance():
    """g2c=0 reduces to a free Zel'dovich draw with variance σ²."""
    key = jax.random.PRNGKey(42)
    res = fem_cascade(
        sigma2=0.1, n_steps=5, N=16, boxsize=1.0, key=key, g2c=0.0,
        cg_tol=1e-6, cg_maxiter=200,
    )
    assert abs(res.sigma2 - 0.1) < 0.03
    # <J> ~ 1 + σ² + O(σ^4); variance of J ~ σ².
    assert abs(float(res.J.mean()) - 1.0) < 0.05
    assert 0.07 < float(jnp.var(res.J)) < 0.15


def test_correction_reduces_skewness_sign():
    """The FEM correction with positive g2c should *decrease* S₃ of J below
    the free Zel'dovich value (~2.0). If it increases, the sign is flipped."""
    key = jax.random.PRNGKey(1)
    free = fem_cascade(sigma2=0.1, n_steps=5, N=16, boxsize=1.0, key=key,
                       g2c=0.0, cg_tol=1e-6, cg_maxiter=200)
    corr = fem_cascade(sigma2=0.1, n_steps=5, N=16, boxsize=1.0, key=key,
                       g2c=0.1, cg_tol=1e-6, cg_maxiter=200)

    def s3(J):
        d = 1.0 / J - 1.0
        return float(jnp.mean(d**3) / jnp.var(d)**1.5)

    s3_free = s3(free.J)
    s3_corr = s3(corr.J)
    # Free Zel'dovich gives S₃ ≈ 2 at this σ². The correction should bring
    # it down toward the RG-flow prediction (~1.4). We allow some slack
    # because N=16 is coarse.
    assert s3_corr < s3_free + 0.05, (
        f"Correction did not reduce S₃: free={s3_free:.3f}, corr={s3_corr:.3f}. "
        "If corr > free, flip the sign of the Hessian in fem_cascade.py."
    )
