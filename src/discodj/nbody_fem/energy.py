"""Variational potential energy and kinetic energy for the FEM N-body.

The discretised gravitational potential energy of the piecewise-uniform
density field is

    U(x) = ½ f(x)ᵀ K(x)⁻¹ f(x) = ½ f(x)ᵀ φ(x),   with K(x)·φ = f(x).

The :func:`potential_energy` below is wrapped in :func:`jax.custom_vjp` so
that ``jax.grad(potential_energy)`` does **not** unroll the CG solve. We use
the **envelope theorem / dual-functional trick**: introduce

    L(x, φ) = φᵀ·f(x) − ½ φᵀ·K(x)·φ.

At the stationary point φ* = K⁻¹f: ``L(x, φ*) = U(x)`` and
``∂L/∂φ|_{φ=φ*} = 0``, so ``∂U/∂x = ∂L/∂x|_{φ=φ*}``. Computing
``∂L/∂x`` with φ held fixed is cheap — one extra assembly + `stencil_matvec`,
**no** adjoint linear solve. This costs ~1.2× the forward, not 2×.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ..fem.assembly import (
    FEMStencil,
    _element_contributions_from_edges,
    assemble_source_tet,
    stencil_matvec,
)
from ..fem.solve import solve_poisson
from .geometry import positions_to_edges


__all__ = ["potential_energy", "kinetic_energy"]


def _solve_phi(x: Array, conn: Array, V_init: Array, q: Array, N: int,
               boxsize: float, rho_bar: float, cg_tol: float,
               cg_maxiter: int) -> tuple[Array, Array]:
    """Shared forward primitive: assemble + solve. Returns ``(φ, f)``.

    ``q`` is the Lagrangian grid; passing it enables Lagrangian-aware
    unwrapping in :func:`positions_to_edges` — robust when individual
    vertices drift beyond ``L/2`` but displacement *differences* between
    connected neighbours stay below ``L/2``.
    """
    edges, V_signed = positions_to_edges(x, conn, boxsize, q=q)
    V_abs = jnp.abs(V_signed)
    rho_tet = rho_bar * V_init / V_abs
    delta_tet = rho_tet / rho_bar - 1.0

    K_e, _, _ = _element_contributions_from_edges(edges)
    K = FEMStencil(K_e=K_e, conn=conn, N=N)
    f = assemble_source_tet(delta_tet, V_abs, conn, N)
    phi = solve_poisson(K, f, tol=cg_tol, maxiter=cg_maxiter, boxsize=boxsize)
    return phi, f


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def potential_energy(
    x: Array,
    conn: Array,
    V_init: Array,
    q: Array,
    N: int,
    boxsize: float,
    rho_bar: float,
    cg_tol: float,
    cg_maxiter: int,
) -> Array:
    """Variational gravitational potential energy of the current mesh.

    :param x: ``(N**3, 3)`` vertex positions.
    :param conn: ``(T, 4)`` Kuhn connectivity (periodic wrap baked in).
    :param V_init: ``(T,)`` initial tet volumes; tet mass is
        ``m_tet = ρ̄ · V_init``, conserved.
    :param N: grid resolution per dimension.
    :param boxsize: side length of the periodic box.
    :param rho_bar: mean density.
    :param cg_tol: forward CG tolerance; also used for the effective adjoint
        via the envelope trick.
    :param cg_maxiter: CG iteration cap.
    :return: scalar ``U = ½ fᵀφ``.
    """
    phi, f = _solve_phi(x, conn, V_init, q, N, boxsize, rho_bar, cg_tol, cg_maxiter)
    return 0.5 * jnp.dot(f, phi)


def _potential_energy_fwd(x, conn, V_init, q, N, boxsize, rho_bar, cg_tol, cg_maxiter):
    # Same input signature as the wrapped potential_energy; nondiff args appear
    # in their original positions (see jax.custom_vjp docs).
    phi, f = _solve_phi(x, conn, V_init, q, N, boxsize, rho_bar, cg_tol, cg_maxiter)
    U = 0.5 * jnp.dot(f, phi)
    return U, (x, conn, V_init, q, phi)


def _potential_energy_bwd(N, boxsize, rho_bar, cg_tol, cg_maxiter, res, g):
    """Backward via envelope theorem: ``∂U/∂x = ∂L/∂x|_{φ=φ*}``.

    ``L(x, φ) = φᵀ f(x) − ½ φᵀ K(x) φ``. We evaluate this as a differentiable
    function of ``x`` with ``φ`` **treated as a constant** (stop_gradient),
    then hand it to :func:`jax.grad`. The resulting gradient costs one
    assembly + one `stencil_matvec` — no adjoint linear solve.
    """
    x, conn, V_init, q, phi = res
    phi_const = jax.lax.stop_gradient(phi)

    def L(x_):
        edges, V_signed = positions_to_edges(x_, conn, boxsize, q=q)
        V_abs = jnp.abs(V_signed)
        rho_tet = rho_bar * V_init / V_abs
        delta_tet = rho_tet / rho_bar - 1.0

        K_e, _, _ = _element_contributions_from_edges(edges)
        K = FEMStencil(K_e=K_e, conn=conn, N=N)
        f = assemble_source_tet(delta_tet, V_abs, conn, N)
        Kphi = stencil_matvec(K, phi_const)
        return jnp.dot(phi_const, f) - 0.5 * jnp.dot(phi_const, Kphi)

    grad_L_x = jax.grad(L)(x)
    # Cotangents for each differentiable input: (x, conn, V_init, q).
    return (g * grad_L_x, jnp.zeros_like(conn), jnp.zeros_like(V_init),
            jnp.zeros_like(q))


potential_energy.defvjp(_potential_energy_fwd, _potential_energy_bwd)


def kinetic_energy(v: Array, masses: Array) -> Array:
    """``T = ½ Σ_i m_i |v_i|²``. Diagnostic only — the integrator doesn't use it."""
    return 0.5 * jnp.sum(masses[:, None] * v * v)
