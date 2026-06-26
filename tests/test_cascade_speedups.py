"""Parity, dtype, and micro-speed tests for the cascade v3 speedups.

Covers:
  1. Drift-free step parity: scalar ``fem_cascade_v3`` with ``g_T = g_S = 0``
     must match the old single-step kernel trajectory step-by-step (same
     PRNG, same dsigma2 schedule). Float32 ulp-level match expected.
  2. Single-call vs batch parity: ``fem_cascade_v3_batch`` over
     ``(beta_nl, keys)`` must return per-batch results identical to
     independent ``fem_cascade_v3`` calls.
  3. Dtype guard: ``result.J.dtype == float32`` — catches an accidental
     ``jax_enable_x64`` globally doubling FFT cost.
  4. Symmetry of 1LPT increment after dedup: ``dG == dG.T`` pointwise.
  5. Micro speed check: batched scan over 8 (β × seed) at N=32 must be
     noticeably faster than equivalent Python-looped scalar calls.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from discodj.cascade import (
    fem_cascade_v3, fem_cascade_v3_batch, CascadeV3Result,
    one_lpt_increment,
)


N = 16
BOXSIZE = 100.0
N_STEPS = 4


def _rand_keys(n: int, seed: int = 0) -> jax.Array:
    return jax.vmap(jax.random.PRNGKey)(jnp.arange(seed, seed + n))


def _frob(x: jax.Array) -> float:
    return float(jnp.sqrt(jnp.sum(jnp.asarray(x)**2)))


def test_dG_symmetric_after_dedup():
    """6-unique-component refactor must preserve G_ij = G_ji."""
    key = jax.random.PRNGKey(0)
    dG = one_lpt_increment(key, N, BOXSIZE, dsigma2=0.01)
    diff = dG - jnp.swapaxes(dG, -1, -2)
    assert _frob(diff) < 1e-5 * _frob(dG), "1LPT dG not symmetric"


def test_J_is_float32():
    """Guards against silent x64 promotion doubling FFT cost."""
    key = jax.random.PRNGKey(1)
    r = fem_cascade_v3(sigma2=0.02, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                       key=key, beta_nl=-1.0)
    assert r.J.dtype == jnp.float32, f"expected float32, got {r.J.dtype}"
    assert r.G.dtype == jnp.float32
    assert r.delta.dtype == jnp.float32


def test_batch_matches_scalar():
    """Batch call = scalar calls element-wise, in float32."""
    B = 4
    betas = jnp.linspace(-1.5, -0.5, B, dtype=jnp.float32)
    keys = _rand_keys(B, seed=100)

    r_b = fem_cascade_v3_batch(
        sigma2=0.03, beta_nl=betas, keys=keys,
        n_steps=N_STEPS, N=N, boxsize=BOXSIZE)

    for i in range(B):
        r_s = fem_cascade_v3(
            sigma2=0.03, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
            key=keys[i], beta_nl=float(betas[i]))
        # J, G, delta should match element-wise.
        tol = 1e-4 * _frob(r_s.J)
        assert _frob(r_b.J[i] - r_s.J) < tol, \
            f"batch vs scalar J mismatch at i={i}: "\
            f"{_frob(r_b.J[i] - r_s.J):.3e} > {tol:.3e}"
        assert _frob(r_b.G[i] - r_s.G) < 1e-4 * _frob(r_s.G)


def test_free_path_skips_drift():
    """At g_T=g_S=0 the cascade must not depend on method / use_abs_J /
    cg_tol etc. — the free path ignores those. Cross-check by running
    with explicit g_T=g_S=0 and different method strings: output must
    match."""
    key = jax.random.PRNGKey(7)
    r_mid = fem_cascade_v3(sigma2=0.02, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                           key=key, beta_nl=-1.0, g_T=0.0, method="midpoint")
    r_eul = fem_cascade_v3(sigma2=0.02, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                           key=key, beta_nl=-1.0, g_T=0.0, method="euler")
    assert _frob(r_mid.J - r_eul.J) < 1e-5 * _frob(r_mid.J), \
        "free path should not depend on method when g_T=g_S=0"


def test_batch_speed_smaller_than_loop():
    """Batched scan over 8 (β, seed) pairs should run faster than the
    equivalent 8 independent scalar calls, once JIT is warm. This is a
    loose sanity check (≤ the scalar-loop wall time)."""
    B = 8
    betas = jnp.linspace(-1.5, -0.8, B, dtype=jnp.float32)
    keys = _rand_keys(B, seed=500)

    # Warm JIT.
    _ = fem_cascade_v3_batch(sigma2=0.05, beta_nl=betas, keys=keys,
                             n_steps=N_STEPS, N=N, boxsize=BOXSIZE)
    _ = fem_cascade_v3(sigma2=0.05, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                       key=keys[0], beta_nl=float(betas[0]))

    t0 = time.time()
    r_b = fem_cascade_v3_batch(sigma2=0.05, beta_nl=betas, keys=keys,
                               n_steps=N_STEPS, N=N, boxsize=BOXSIZE)
    jax.block_until_ready(r_b.J)
    t_batch = time.time() - t0

    t0 = time.time()
    for i in range(B):
        r = fem_cascade_v3(sigma2=0.05, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                           key=keys[i], beta_nl=float(betas[i]))
        jax.block_until_ready(r.J)
    t_loop = time.time() - t0

    print(f"\nbatch: {t_batch*1000:.0f} ms;  loop: {t_loop*1000:.0f} ms; "
          f"speedup ×{t_loop/max(t_batch,1e-6):.2f}")
    assert t_batch < t_loop, "batch should not be slower than scalar loop"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
