"""Sanity test for the per-tet deformation tensor + invariants."""
import jax
import jax.numpy as jnp
import numpy as np

from discodj.analysis import (
    load_sim, kuhn_cube_volumes,
    kuhn_tet_deformation_tensor_from_positions,
    strain_invariants_from_F,
)
from discodj.analysis.phase_space_sheet import _eulerian_positions


def test_detF_matches_cube_sum():
    """For each cube, Σ_t det(F_t) · (d_lag³ / 6) == V_cube."""
    snap = load_sim(128, "nocutoff", a=0.10)
    x_np, L, N = _eulerian_positions(snap)
    x = jnp.asarray(x_np)
    stride = 4
    N_c = N // stride
    d_lag = stride * L / N

    F = kuhn_tet_deformation_tensor_from_positions(
        x, L, stride=stride, overlapping=False)     # (6, N_c, N_c, N_c, 3, 3)
    detF = jnp.linalg.det(F)                        # (6, N_c, N_c, N_c)
    # Per-cube signed volume = (d_lag³/6) · Σ_t detF
    V_cube_from_F = (d_lag**3 / 6.0) * detF.sum(axis=0)
    V_cube_ref = kuhn_cube_volumes(snap, stride=stride, overlapping=False)
    diff = float(jnp.max(jnp.abs(V_cube_from_F - jnp.asarray(V_cube_ref))))
    ref  = float(jnp.max(jnp.abs(jnp.asarray(V_cube_ref))))
    print(f"  max |V_F − V_cube| / max |V| = {diff/ref:.2e}")
    assert diff / ref < 1e-5


def test_invariants_trace_identity():
    """Algebraic identities among the invariants: I_1² − 2·I_2 = tr(G²)."""
    snap = load_sim(128, "nocutoff", a=0.10)
    x_np, L, N = _eulerian_positions(snap)
    x = jnp.asarray(x_np)
    F = kuhn_tet_deformation_tensor_from_positions(x, L, stride=4,
                                                     overlapping=False)
    inv = strain_invariants_from_F(F)
    # Check the identity used in I_2's definition
    resid = float(jnp.max(jnp.abs(inv["I1"]**2 - 2 * inv["I2"] - inv["trG2"])))
    print(f"  max |I_1² − 2 I_2 − tr(G²)| = {resid:.2e}")
    assert resid < 1e-3
    # Σ is traceless by construction
    print(f"  I_1 range = [{float(inv['I1'].min()):.2f}, {float(inv['I1'].max()):.2f}]")
    print(f"  tr(Σ²) ≥ 0 min = {float(inv['trS2'].min()):.2e}")
    assert float(inv["trS2"].min()) >= -1e-4        # positive up to roundoff


if __name__ == "__main__":
    test_detF_matches_cube_sum()
    test_invariants_trace_identity()
    print("\nAll invariant checks passed.")
