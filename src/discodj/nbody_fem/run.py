"""Top-level driver for the FEM N-body solver."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ..fem.kuhn import kuhn_connectivity
from .cosmo import build_time_table, cosmological_kdk_step
from .energy import potential_energy, kinetic_energy
from .force import force_fn as _force_fn
from .geometry import grid_positions, positions_to_edges
from .integrator import NBodyState, kdk_step, leapfrog_run


__all__ = ["NBodyResult", "fem_nbody", "fem_nbody_cosmo"]


@dataclass
class NBodyResult:
    state: NBodyState
    log: list   # whatever the monitor produced, step-by-step


def fem_nbody(
    x0: Array,
    v0: Array,
    dt: float,
    n_steps: int,
    boxsize: float,
    rho_bar: float = 1.0,
    monitor_every: int = 0,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
    check_shell_crossing: bool = True,
) -> NBodyResult:
    """Run static-Newtonian FEM N-body starting from ``(x0, v0)``.

    :param x0: initial positions ``(N**3, 3)``; assumed flattened over the
        Kuhn connectivity's C-order.
    :param v0: initial velocities ``(N**3, 3)``.
    :param dt: timestep.
    :param n_steps: number of KDK steps.
    :param boxsize: periodic box side.
    :param rho_bar: mean density (tet masses = ρ̄ · V_init).
    :param monitor_every: if >0, record (step, energy diagnostics, min V_tet)
        every N steps. 0 → no monitoring.
    :param cg_tol: Poisson CG tolerance.
    :param cg_maxiter: Poisson CG iteration cap.
    :param check_shell_crossing: if True, the monitor raises
        ``RuntimeError`` when any tet inverts; if False, just records
        ``min(V_signed)``.
    """
    n_points = x0.shape[0]
    N = round(n_points ** (1 / 3))
    assert N**3 == n_points, f"x0 has {n_points} points; not a perfect cube"

    conn = kuhn_connectivity(N)

    # Tet mass = ρ̄ · V_Lagrangian, evaluated on the *undeformed* grid. The
    # caller may pass a perturbed x0 (1LPT ICs), but the Lagrangian mass of
    # each element is set by the uniform regular grid — so δ_tet at t=0 is
    # driven by the deformation, not canceled by it.
    q = grid_positions(N, boxsize)
    _, V0_signed = positions_to_edges(q, conn, boxsize, q=q)
    V_init = jnp.abs(V0_signed)

    # Vertex masses (for KE diagnostic): lumped mass = Σ_{tets∋i} ρ̄·V_tet/4.
    masses_per_tet_vertex = jnp.broadcast_to(
        (rho_bar * V_init / 4.0)[:, None], (conn.shape[0], 4)
    )
    masses = jax.ops.segment_sum(
        masses_per_tet_vertex.reshape(-1), conn.reshape(-1), num_segments=n_points
    )

    force_closure = lambda x: _force_fn(  # noqa: E731
        x, conn, V_init, q, N, boxsize, rho_bar, cg_tol, cg_maxiter,
    )

    def monitor(step, state):
        _, V_signed = positions_to_edges(state.x, conn, boxsize, q=q)
        min_V = float(V_signed.min())
        if check_shell_crossing and min_V <= 0:
            raise RuntimeError(
                f"Shell crossing at step {step}: {int((V_signed <= 0).sum())} "
                f"tets inverted (min V_signed = {min_V:.3e})."
            )
        T = float(kinetic_energy(state.v, masses))
        U = float(
            potential_energy(state.x, conn, V_init, q, N, boxsize, rho_bar,
                             cg_tol, cg_maxiter)
        )
        return {"step": step, "T": T, "U": U, "E": T + U, "min_V": min_V}

    state_final, log = leapfrog_run(
        state0=NBodyState(x=x0, v=v0),
        dt=dt,
        n_steps=n_steps,
        force_fn=force_closure,
        masses=masses,
        boxsize=boxsize,
        monitor_every=monitor_every,
        monitor_fn=monitor if monitor_every > 0 else None,
    )
    return NBodyResult(state=state_final, log=log)


def fem_nbody_cosmo(
    x0: Array,
    p0: Array,
    cosmo,
    a_ini: float,
    a_end: float,
    n_steps: int,
    boxsize: float,
    rho_bar: float = 1.0,
    spacing: str = "log_a",
    monitor_every: int = 0,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
    check_shell_crossing: bool = True,
) -> NBodyResult:
    """Cosmological FEM N-body with FastPM-style D-time DKD.

    :param x0: comoving positions at ``a_ini`` ``(N**3, 3)``.
    :param p0: D-time momentum ``p = dx/dD`` at ``a_ini`` ``(N**3, 3)``. For
        ZA ICs this is exactly ``ψ`` — constant with ``a``.
    :param cosmo: DISCO-DJ ``Cosmology`` (with ``with_timetables()`` called).
    :param a_ini: starting scale factor.
    :param a_end: ending scale factor.
    :param n_steps: number of KDK steps.
    :param boxsize: comoving box size (same units as ``x0``).
    :param rho_bar: background density (code units). Set so that tet masses
        ``m_tet = ρ̄·V_lagr`` give the intended total mass — for cosmology
        we typically keep ``rho_bar=1`` and absorb any scaling in ``F_scale``.
    :param spacing: step spacing — ``"log_a"`` (default), ``"a"``, or ``"D"``.
    """
    n_points = x0.shape[0]
    N = round(n_points ** (1 / 3))
    assert N**3 == n_points

    conn = kuhn_connectivity(N)
    q = grid_positions(N, boxsize)
    _, V0_signed = positions_to_edges(q, conn, boxsize, q=q)
    V_init = jnp.abs(V0_signed)

    force_closure = lambda x: _force_fn(  # noqa: E731
        x, conn, V_init, q, N, boxsize, rho_bar, cg_tol, cg_maxiter,
    )

    coeffs = build_time_table(cosmo, a_ini, a_end, n_steps, boxsize, N,
                              spacing=spacing)

    def monitor(step, x, p, a):
        _, V_signed = positions_to_edges(x, conn, boxsize, q=q)
        min_V = float(V_signed.min())
        if check_shell_crossing and min_V <= 0:
            raise RuntimeError(
                f"Shell crossing at step {step} (a={a:.3f}): "
                f"{int((V_signed <= 0).sum())} inverted tets."
            )
        # Minimum-image displacement so wraps near the box edge don't read as
        # ~L jumps. (Real shell crossing is caught separately via min_V.)
        dx = x - q
        dx = dx - boxsize * jnp.round(dx / boxsize)
        psi_mean = float(jnp.mean(jnp.linalg.norm(dx, axis=-1)))
        p_mean = float(jnp.mean(jnp.linalg.norm(p, axis=-1)))
        return {"step": step, "a": a, "min_V": min_V,
                "psi_mean": psi_mean, "p_mean": p_mean}

    x, p = x0, p0
    log = []
    if monitor_every > 0:
        log.append(monitor(0, x, p, a_ini))
    for step in range(n_steps):
        x, p = cosmological_kdk_step(x, p, force_closure, coeffs, step, boxsize)
        if monitor_every > 0 and (step + 1) % monitor_every == 0:
            log.append(monitor(step + 1, x, p, float(coeffs["a_edges"][step + 1])))

    return NBodyResult(state=NBodyState(x=x, v=p), log=log)
