"""Kick-drift-kick leapfrog integrator.

Static-Newtonian phase only; the cosmological (BullFrog-style) splitting
goes in phase 2.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


__all__ = ["NBodyState", "kdk_step", "leapfrog_run"]


class NBodyState(NamedTuple):
    """Phase-space state: positions + velocities."""

    x: Array  # (N³, 3)
    v: Array  # (N³, 3)


def kdk_step(state: NBodyState, dt: float, force_fn: Callable[[Array], Array],
             masses: Array, boxsize: float) -> NBodyState:
    """Kick-Drift-Kick leapfrog step.

    ``force_fn`` must be a closure that takes positions ``x`` and returns the
    force array (i.e., already parametrised with conn/V_init/N/etc.).

    :param state: incoming ``(x, v)``.
    :param dt: timestep.
    :param force_fn: ``x -> F``. Typically
        ``functools.partial(force, conn=conn, V_init=V_init, N=N, boxsize=boxsize, …)``.
    :param masses: per-vertex masses ``(N**3,)``.
    :param boxsize: box side length for periodic wrapping after the drift.
    """
    F = force_fn(state.x)
    inv_m = 1.0 / masses[:, None]

    # First half-kick
    v_half = state.v + 0.5 * dt * F * inv_m
    # Drift (positions kept unwrapped — neighbours only see each other via
    # the Lagrangian-aware unwrapping inside ``positions_to_edges``).
    x_new = state.x + dt * v_half
    # Second kick uses the new force
    F_new = force_fn(x_new)
    v_new = v_half + 0.5 * dt * F_new * inv_m
    return NBodyState(x=x_new, v=v_new)


def leapfrog_run(state0: NBodyState, dt: float, n_steps: int,
                 force_fn: Callable[[Array], Array], masses: Array,
                 boxsize: float, monitor_every: int = 0,
                 monitor_fn: Callable | None = None) -> tuple[NBodyState, list]:
    """Drive ``n_steps`` of KDK leapfrog.

    Uses a Python-level loop (not ``jax.lax.scan``) to make ``monitor_fn``
    ergonomic and the error messages on shell-crossing clean. Each ``kdk_step``
    is individually JIT-compiled via ``force_fn``, so the per-step overhead
    is just one dispatch.

    :param monitor_every: if >0, call ``monitor_fn(step, state)`` every N
        steps. The returned list collects whatever the monitor returns.
    """
    state = state0
    log = []
    if monitor_fn is not None and monitor_every > 0:
        log.append(monitor_fn(0, state))

    for step in range(1, n_steps + 1):
        state = kdk_step(state, dt, force_fn, masses, boxsize)
        if monitor_fn is not None and monitor_every > 0 and step % monitor_every == 0:
            log.append(monitor_fn(step, state))

    return state, log
