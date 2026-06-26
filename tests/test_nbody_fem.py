"""Phase 1 (static Newtonian) tests for the FEM N-body solver."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from discodj.fem.kuhn import kuhn_connectivity
from discodj.nbody_fem import (
    grid_positions,
    positions_to_edges,
    potential_energy,
    force_fn,
    fem_nbody,
)


def _make_ics(N, L, amp=0.02):
    conn = kuhn_connectivity(N)
    x0 = grid_positions(N, L)
    x_pert = x0.at[:, 0].set(x0[:, 0] + amp * L * jnp.sin(2 * jnp.pi * x0[:, 0] / L))
    q = grid_positions(N, L)
    _, V_signed = positions_to_edges(q, conn, L, q=q)
    V_init = jnp.abs(V_signed)
    return conn, x0, x_pert, V_init, q


# ---------------------------------------------------------- test 1


@pytest.mark.parametrize("N", [8, 16])
def test_zero_perturbation_stationary(N):
    """``x0 = grid``, ``v0 = 0`` → no motion, no force, U=0."""
    L = 1.0
    conn, x0, _, V_init, q = _make_ics(N, L)

    U = potential_energy(x0, conn, V_init, q, N, L, 1.0, 1e-8, 20)
    F = force_fn(x0, conn, V_init, q, N, L, 1.0, 1e-8, 20)
    assert float(jnp.abs(U)) < 1e-10
    assert float(jnp.max(jnp.abs(F))) < 1e-6

    res = fem_nbody(x0, jnp.zeros_like(x0), dt=0.01, n_steps=20,
                    boxsize=L, rho_bar=1.0, monitor_every=0)
    assert float(jnp.max(jnp.abs(res.state.x - x0))) < 1e-6
    assert float(jnp.max(jnp.abs(res.state.v))) < 1e-6


# ---------------------------------------------------------- test 2


def test_force_is_gradient_of_U():
    """Finite-difference check that ``F = -∂U/∂x`` in the x-direction of a
    sine-wave perturbation (symmetry breaks y/z components, which are
    numerical noise in the FD and exactly zero in AD — we only compare
    the x-component)."""
    N, L = 8, 1.0
    conn, x0, x_pert, V_init, q = _make_ics(N, L)
    F = force_fn(x_pert, conn, V_init, q, N, L, 1.0, 1e-9, 50)

    # Check a handful of particles at non-zero perturbation locations.
    eps = 1e-4
    for i in (64, 128, 192, 320):  # (ix=1,2,3,5)
        x_ad = float(F[i, 0])
        xp = x_pert.at[i, 0].add(eps)
        xm = x_pert.at[i, 0].add(-eps)
        Up = float(potential_energy(xp, conn, V_init, q, N, L, 1.0, 1e-10, 100))
        Um = float(potential_energy(xm, conn, V_init, q, N, L, 1.0, 1e-10, 100))
        x_fd = -(Up - Um) / (2 * eps)
        # Both should be far from zero, and within ~10% of each other
        # (float32 precision + CG tolerance in U).
        assert abs(x_ad) > 1e-7, f"AD force at i={i} unexpectedly tiny: {x_ad:.3e}"
        rel = abs(x_ad - x_fd) / (abs(x_fd) + 1e-30)
        assert rel < 0.15, (
            f"F_ad={x_ad:.3e}  F_fd={x_fd:.3e}  rel err={rel:.3f} at i={i}"
        )


# ---------------------------------------------------------- test 3


def test_energy_conservation_moderate_drift():
    """Run leapfrog for 50 steps with the variational force.

    Note: we expect *bounded* drift, not machine-precision — CG is only
    solved to tol=1e-7 so each force eval has ~1e-4 relative error; over 50
    steps that compounds to O(percent) in ΔE/E, which is fine. If we need
    tighter we lower cg_tol.
    """
    N, L = 16, 1.0
    _, _, x_pert, V_init, _ = _make_ics(N, L, amp=0.02)
    v0 = jnp.zeros_like(x_pert)

    res = fem_nbody(x_pert, v0, dt=0.01, n_steps=50, boxsize=L,
                    rho_bar=1.0, monitor_every=5, cg_tol=1e-9, cg_maxiter=30)

    E0 = res.log[0]["E"]
    E_final = res.log[-1]["E"]
    dE_rel = abs(E_final - E0) / abs(E0)
    assert dE_rel < 0.02, f"|ΔE/E|={dE_rel:.3e} at step 50"
    # Also: min volume should stay > 0 (no shell crossing at these settings).
    assert res.log[-1]["min_V"] > 0


# ---------------------------------------------------------- test 4


def test_detects_shell_crossing():
    """An initial configuration with an inverted tet → driver raises."""
    N, L = 8, 1.0
    conn = kuhn_connectivity(N)
    x0 = grid_positions(N, L)
    # Swap two vertices across an edge of a single cube — guaranteed to
    # invert at least one tet (the 4 containing vertex 0 of the Kuhn cube).
    # We pick vertex (0,0,0) and swap its position with its x-neighbour (1,0,0);
    # that puts them in each other's Lagrangian slots, inverting the tets
    # straddling that edge.
    h = L / N
    x_swap = x0.at[0].set(x0[1]).at[1].set(x0[0])
    v0 = jnp.zeros_like(x_swap)

    with pytest.raises(RuntimeError, match="Shell crossing"):
        fem_nbody(x_swap, v0, dt=0.005, n_steps=2, boxsize=L,
                  rho_bar=1.0, monitor_every=1)


# =========================================================== Phase 2/3


def test_cosmological_growth_matches_D_plus():
    """A pure 1LPT initial condition should evolve as ψ(a) = D(a)/D(a_ini)·ψ(a_ini)."""
    from discodj import DiscoDJ
    from discodj.nbody_fem import fem_nbody_cosmo

    N, L = 16, 250.0
    dj = DiscoDJ(dim=3, res=N, boxsize=L).with_timetables()
    cosmo = dj.cosmo

    q = grid_positions(N, L)
    ksine = 2 * jnp.pi / L
    amp = 0.5
    psi = jnp.stack([amp * jnp.sin(ksine * q[:, 0]),
                     jnp.zeros(N**3), jnp.zeros(N**3)], axis=-1)
    a_ini, a_end = 0.1, 0.5
    D_ini = float(cosmo.Dplus(a_ini))
    D_end = float(cosmo.Dplus(a_end))
    target = D_end / D_ini

    res = fem_nbody_cosmo(q + D_ini * psi, psi, cosmo, a_ini, a_end,
                          n_steps=10, boxsize=L, monitor_every=1,
                          cg_tol=1e-8, cg_maxiter=15)

    # Ratio of final / initial minimum-image displacement magnitude.
    dx_final = res.state.x - q
    dx_final = dx_final - L * jnp.round(dx_final / L)
    psi_final_mean = float(jnp.mean(jnp.linalg.norm(dx_final, axis=-1)))
    psi_init_mean = float(jnp.mean(jnp.linalg.norm(D_ini * psi, axis=-1)))
    observed = psi_final_mean / psi_init_mean

    rel_err = abs(observed - target) / target
    assert rel_err < 0.05, (
        f"Observed {observed:.3f}, expected {target:.3f} (rel err {rel_err:.3%})"
    )


def test_matches_discodj_pm_at_matched_ics():
    """At low σ² (pre-shell-crossing), FEM N-body and DiscoDJ PM should
    produce almost-identical trajectories when handed the same GRF
    initial conditions."""
    from discodj import DiscoDJ
    from discodj.nbody_fem import fem_nbody_cosmo, ics_from_discodj

    N, L = 16, 250.0
    a_ini, a_end, n_steps = 0.05, 0.1, 10

    dj = (DiscoDJ(dim=3, res=N, boxsize=L)
          .with_timetables()
          .with_linear_ps()
          .with_ics(seed=42, white_noise_space="fourier")
          .with_lpt(n_order=1))

    x0, p0 = ics_from_discodj(dj, a_ini=a_ini)
    res_fem = fem_nbody_cosmo(x0, p0, dj.cosmo, a_ini, a_end, n_steps=n_steps,
                              boxsize=L, cg_tol=1e-8, cg_maxiter=15)
    x_pm, _, _ = dj.run_nbody(a_ini=a_ini, a_end=a_end, n_steps=n_steps,
                              stepper="fastpm", method="pm", res_pm=N,
                              time_var="log_a", ic_method="lpt",
                              nlpt_order_ics=1)

    x_pm = np.asarray(x_pm).reshape(N**3, 3)
    dx = np.asarray(res_fem.state.x) - x_pm
    dx = dx - L * np.round(dx / L)
    per_part = np.linalg.norm(dx, axis=-1)

    cell = L / N
    # At this redshift range the system is deeply linear — trajectories should
    # agree to ~1% of a cell in the median. (First measurement: 0.27%.)
    median = float(np.median(per_part))
    p95 = float(np.percentile(per_part, 95))
    assert median < 0.05 * cell, f"median |Δx| = {median:.4f} Mpc/h ({median/cell:.3f} cells)"
    assert p95 < 0.10 * cell, f"p95 |Δx| = {p95:.4f} Mpc/h ({p95/cell:.3f} cells)"
