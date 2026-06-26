"""Differentiability check for the JAX-jitted PSS volume kernel."""
import numpy as np
import jax
import jax.numpy as jnp

from discodj.analysis import (load_sim, kuhn_cube_volumes_from_positions)
from discodj.analysis.phase_space_sheet import _eulerian_positions


def test_grad_through_kuhn_volumes():
    """Gradients of ⟨|V_cube|⟩ w.r.t. particle positions should be finite."""
    snap = load_sim(128, "nocutoff", a=0.10)
    x_np, L, N = _eulerian_positions(snap)
    x = jnp.asarray(x_np)

    def loss(x):
        V = kuhn_cube_volumes_from_positions(x, L, stride=4, overlapping=False)
        return jnp.sum(V**2)             # scalar functional of all cubes

    grad_fn = jax.grad(loss)
    g = grad_fn(x)
    jax.block_until_ready(g)
    g_np = np.asarray(g)
    print(f"  grad shape       : {g_np.shape}")
    print(f"  grad max abs     : {np.abs(g_np).max():.3e}")
    print(f"  grad mean abs    : {np.abs(g_np).mean():.3e}")
    assert np.isfinite(g_np).all(), "non-finite gradient values"
    assert np.abs(g_np).max() > 0, "gradient is identically zero"


def test_jvp_is_consistent():
    """JVP gives the same result as finite differences for a small perturbation."""
    snap = load_sim(128, "nocutoff", a=0.10)
    x_np, L, N = _eulerian_positions(snap)
    x = jnp.asarray(x_np)
    # Small perturbation direction
    rng = np.random.default_rng(0)
    dx_np = rng.normal(size=x_np.shape).astype(np.float32) * 1e-3
    dx = jnp.asarray(dx_np)

    def f(x):
        return kuhn_cube_volumes_from_positions(x, L, stride=4, overlapping=False)

    # JVP
    _, V_dot = jax.jvp(f, (x,), (dx,))
    # Finite difference (central)
    V_plus  = f(x + 0.5 * dx)
    V_minus = f(x - 0.5 * dx)
    fd = V_plus - V_minus
    rel = float(jnp.max(jnp.abs(V_dot - fd)) / jnp.max(jnp.abs(V_dot)))
    print(f"  JVP vs FD rel error: {rel:.3e}   (float32 FD noise ~1e-2)")
    assert rel < 2e-2           # float32 tolerance, O(1e-3) step


if __name__ == "__main__":
    test_grad_through_kuhn_volumes()
    test_jvp_is_consistent()
    print("\nAll differentiability checks passed.")
