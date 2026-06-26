"""Force per particle: ``F_i = -∂U/∂q_i`` via :mod:`jax.grad` + custom VJP."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .energy import potential_energy


__all__ = ["force_fn"]


@partial(jax.jit, static_argnames=("N", "boxsize", "rho_bar", "cg_tol", "cg_maxiter"))
def force_fn(
    x: Array,
    conn: Array,
    V_init: Array,
    q: Array,
    N: int,
    boxsize: float,
    rho_bar: float = 1.0,
    cg_tol: float = 1e-7,
    cg_maxiter: int = 10,
) -> Array:
    """Gravitational force per vertex (``(N**3, 3)``).

    :param x: ``(N**3, 3)`` vertex positions.
    :param conn: ``(T, 4)`` Kuhn connectivity.
    :param V_init: ``(T,)`` initial tet volumes.
    :param N: grid resolution per dimension.
    :param boxsize: side length of the periodic box.
    :param rho_bar: mean density.
    :param cg_tol: Poisson CG tolerance.
    :param cg_maxiter: Poisson CG iteration cap.
    :return: force array of shape ``(N**3, 3)`` — ``F_i = -∂U/∂x_i``.
    """
    grad_U = jax.grad(potential_energy, argnums=0)(
        x, conn, V_init, q, N, boxsize, rho_bar, cg_tol, cg_maxiter
    )
    return -grad_U
