"""Geometric kernels for the FEM N-body: positions → tet edges, volumes, ρ."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


__all__ = ["positions_to_edges", "tet_densities", "grid_positions"]


def grid_positions(N: int, boxsize: float) -> Array:
    """Regular Lagrangian grid positions, shape ``(N**3, 3)``.

    Matches DISCO-DJ's convention: ``q[i,j,k] = (i·h, j·h, k·h)`` with
    ``h = L/N``; flattened in ``(ix, iy, iz)`` C-order so the ``conn``
    table from :func:`discodj.fem.kuhn.kuhn_connectivity` indexes into it
    directly.
    """
    h = boxsize / N
    ix, iy, iz = jnp.meshgrid(
        jnp.arange(N), jnp.arange(N), jnp.arange(N), indexing="ij"
    )
    return jnp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1).astype(
        jnp.float32
    ) * h


def _minimum_image(dx: Array, boxsize: float) -> Array:
    """Wrap ``dx`` into ``[-L/2, L/2)`` per axis."""
    return dx - boxsize * jnp.round(dx / boxsize)


def positions_to_edges(x: Array, conn: Array, boxsize: float,
                       q: Array | None = None) -> tuple[Array, Array]:
    """Build deformed tet edges and signed volumes from vertex positions.

    :param x: ``(N**3, 3)`` vertex positions. Preferably *unwrapped* — we
        don't require positions in ``[0, L)``.
    :param conn: ``(T, 4)`` connectivity; ``T = 6 N**3``.
    :param boxsize: side length of the periodic box.
    :param q: optional Lagrangian grid positions ``(N**3, 3)``. If provided,
        edges are built as ``(q_k − q_0) + minimum_image(ψ_k − ψ_0)`` with
        ``ψ = x − q``. This is the Lagrangian-aware unwrapping: it's exact
        so long as no displacement *difference* between neighbouring vertices
        exceeds ``L/2``, which is the true limit for a simplicial mesh to
        stay well-defined.

        If ``q`` is ``None`` (legacy path) we fall back to minimum-image on
        raw position differences. That silently breaks when any single
        vertex has drifted more than ``L/2``.
    :return: ``(edges, V_signed)`` with shapes ``(T, 3, 3)`` and ``(T,)``.
    """
    if q is None:
        v0, v1, v2, v3 = (x[conn[:, k]] for k in range(4))
        e1 = _minimum_image(v1 - v0, boxsize)
        e2 = _minimum_image(v2 - v0, boxsize)
        e3 = _minimum_image(v3 - v0, boxsize)
    else:
        psi = x - q
        q0, q1, q2, q3 = (q[conn[:, k]] for k in range(4))
        p0, p1, p2, p3 = (psi[conn[:, k]] for k in range(4))
        # q differences use the exact Kuhn geometry; the periodic wrap here
        # is just to handle the boundary tets that reach across the box.
        e1 = _minimum_image(q1 - q0, boxsize) + _minimum_image(p1 - p0, boxsize)
        e2 = _minimum_image(q2 - q0, boxsize) + _minimum_image(p2 - p0, boxsize)
        e3 = _minimum_image(q3 - q0, boxsize) + _minimum_image(p3 - p0, boxsize)

    edges = jnp.stack([e1, e2, e3], axis=1)
    V_signed = jnp.einsum("ti,ti->t", e1, jnp.cross(e2, e3)) / 6.0
    return edges, V_signed


def tet_densities(V_abs: Array, rho_bar: float, V_init: Array) -> Array:
    """Compute ``ρ_tet = ρ̄ · V_init / V_abs`` (mass-conserving).

    :param V_abs: ``(T,)`` current |signed tet volumes|.
    :param rho_bar: mean density.
    :param V_init: ``(T,)`` initial |tet volumes| (constants of motion).
    :return: per-tet density.
    """
    return rho_bar * V_init / V_abs
