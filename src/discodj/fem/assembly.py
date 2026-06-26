"""Stiffness matrix and RHS assembly for P1 FEM on the Kuhn tessellation.

Given a deformation gradient ``G = d psi / d q`` sampled at each vertex of a
periodic N^3 grid, we build the deformed Kuhn mesh (using the per-tet average
``G`` to displace edges) and assemble:

- the global P1 stiffness matrix ``K phi = f`` (BCOO sparse),
- the lumped-mass source vector ``f_i = sum_{tets ∋ i} δ_i · |V_tet|/4``
  where ``δ = 1/det(I+G) − 1`` at each vertex.

Everything is a pure JAX function — the outer loop over tets is a ``vmap``,
and the scatter-add uses :func:`jax.ops.segment_sum`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from .kuhn import kuhn_connectivity, lagrangian_edge_vectors


__all__ = [
    "assemble_stiffness",
    "assemble_stiffness_stencil",
    "assemble_stiffness_stencil_from_edges",
    "assemble_source",
    "assemble_source_tet",
    "FEMStencil",
    "stencil_matvec",
    "stencil_diag",
]


class FEMStencil(NamedTuple):
    """Matrix-free FEM stiffness operator.

    ``K_e`` holds per-tet 4×4 element stiffness matrices (shape ``(T, 4, 4)``)
    and ``conn`` the tet-to-vertex index table. Applying the operator via
    :func:`stencil_matvec` — gather ``x`` at the 4 local vertices, multiply by
    ``K_e``, scatter-add with :func:`jax.ops.segment_sum` — avoids the
    ``jax.experimental.sparse`` BCOO matmul, which on CPU XLA serializes on
    atomic scatters.  Scales across cores because the batched 4×4 matmul and
    segment_sum both parallelise cleanly.
    """

    K_e: Array  # (T, 4, 4)
    conn: Array  # (T, 4) int32
    N: int

    def __matmul__(self, x: Array) -> Array:
        return stencil_matvec(self, x)


def stencil_matvec(stencil: FEMStencil, x: Array) -> Array:
    """Apply a :class:`FEMStencil` operator to a vector ``x`` of shape ``(N³,)``."""
    N = stencil.N
    K_e = stencil.K_e
    conn = stencil.conn
    # Gather local vertex values (T, 4).
    x_tet = x[conn]
    # Batched 4×4 matmul -> per-tet contribution (T, 4).
    y_tet = jnp.einsum("tij,tj->ti", K_e, x_tet)
    # Scatter-add to N³ global vector.
    return jax.ops.segment_sum(y_tet.ravel(), conn.ravel(), num_segments=N**3)


def stencil_diag(stencil: FEMStencil) -> Array:
    """Diagonal of a stencil operator (sum of self-contributions per vertex)."""
    N = stencil.N
    K_e = stencil.K_e
    conn = stencil.conn
    # Each tet contributes K_e[i, i] at its ith local vertex to that vertex's diagonal.
    diag_local = jnp.stack([K_e[:, i, i] for i in range(4)], axis=1)  # (T, 4)
    return jax.ops.segment_sum(
        diag_local.ravel(), conn.ravel(), num_segments=N**3
    )


def _tet_vertex_gather(field_flat: Array, conn: Array) -> Array:
    """Gather ``field_flat`` at each tet's four vertices -> ``(T, 4, ...)``."""
    return field_flat[conn]


def _element_contributions_from_edges(edges_def: Array) -> tuple[Array, Array, Array]:
    """Compute per-tet stiffness, |V|, and cotan vectors from deformed edges.

    :param edges_def: ``(T, 3, 3)`` — for each tet the three edge vectors
        ``e_k = v_{k+1} − v_0`` in its *deformed* geometry. This is the
        single shared input of the FEM geometry layer; the cascade path
        builds ``edges_def = (I + G_avg) @ e_lag`` and the N-body path
        builds them directly from deformed vertex positions.
    :return: ``(K_e, V_abs, c)`` with
        - ``K_e`` shape ``(T, 4, 4)`` element stiffness
        - ``V_abs`` shape ``(T,)`` |signed volume|
        - ``c`` shape ``(T, 4, 3)`` face-area cotan vectors (opposite each
          local vertex). Gradients of P1 basis: ``∇λ_i = −c_i/(6V)``.
    """
    e1 = edges_def[:, 0, :]
    e2 = edges_def[:, 1, :]
    e3 = edges_def[:, 2, :]

    e2xe3 = jnp.cross(e2, e3)
    V = jnp.einsum("ti,ti->t", e1, e2xe3) / 6.0
    V_abs = jnp.abs(V)

    # Deformed vertex positions relative to v0.
    v0 = jnp.zeros_like(e1)
    v1, v2, v3 = v0 + e1, v0 + e2, v0 + e3

    # Cotangent-like face-area vectors (= 2× outward area normal, opposite each vertex).
    c0 = jnp.cross(v2 - v1, v3 - v1)
    c1 = jnp.cross(v3 - v0, v2 - v0)
    c2 = jnp.cross(v1 - v0, v3 - v0)
    c3 = -(c0 + c1 + c2)
    c = jnp.stack([c0, c1, c2, c3], axis=1)  # (T, 4, 3)

    K_e = jnp.einsum("tid,tjd->tij", c, c) / (36.0 * V_abs[:, None, None])
    return K_e, V_abs, c


def _edges_from_G(G_vertices_flat: Array, conn: Array, lag_edges: Array) -> Array:
    """Cascade path: deformed edges from vertex-averaged G × Lagrangian edges.

    :param G_vertices_flat: ``(N**3, 3, 3)`` G at vertices.
    :param conn: ``(T, 4)`` connectivity.
    :param lag_edges: ``(6, 3, 3)`` undeformed edges per tet type.
    :return: ``(T, 3, 3)`` deformed edges.
    """
    T = conn.shape[0]
    n_cubes = T // 6
    edges_lag = jnp.broadcast_to(lag_edges[None, :, :, :], (n_cubes, 6, 3, 3))
    edges_lag = edges_lag.reshape(T, 3, 3)

    G_tet = _tet_vertex_gather(G_vertices_flat, conn).mean(axis=1)
    F = jnp.eye(3) + G_tet
    return jnp.einsum("tij,tkj->tki", F, edges_lag)


def _element_contributions(G_vertices_flat: Array, conn: Array,
                           lag_edges: Array) -> tuple[Array, Array]:
    """Back-compat wrapper: compute per-tet ``(K_e, |V|)`` from cascade G-input.

    Kept as a thin shim so existing call sites (the cascade's
    :func:`assemble_stiffness_stencil` and :func:`assemble_stiffness`) don't
    need to change.
    """
    edges_def = _edges_from_G(G_vertices_flat, conn, lag_edges)
    K_e, V_abs, _ = _element_contributions_from_edges(edges_def)
    return K_e, V_abs


def assemble_stiffness_stencil(G: Array, boxsize: float) -> FEMStencil:
    """Assemble the FEM stiffness as a matrix-free :class:`FEMStencil` operator.

    Prefer this over :func:`assemble_stiffness` — the stencil form parallelises
    across CPU cores where BCOO matmul serialises.
    """
    N = G.shape[0]
    assert G.shape == (N, N, N, 3, 3), f"G has shape {G.shape}, expected (N,N,N,3,3)"

    G_flat = G.reshape(N**3, 3, 3)
    conn = kuhn_connectivity(N)
    lag_edges = lagrangian_edge_vectors(N, boxsize)

    K_e, _ = _element_contributions(G_flat, conn, lag_edges)
    return FEMStencil(K_e=K_e, conn=conn, N=N)


def assemble_stiffness(G: Array, boxsize: float) -> BCOO:
    """Assemble the global FEM stiffness matrix ``K`` on the deformed mesh.

    Use :func:`assemble_stiffness_stencil` instead unless you specifically
    need a BCOO matrix (e.g. for SciPy interop): the stencil form is
    materially faster inside CG on CPU.

    :param G: deformation gradient at each Lagrangian vertex, shape
        ``(N, N, N, 3, 3)``. The ``(d, g)`` convention matches
        :meth:`DiscoDJ.evaluate_jacobian_from_psi`: the last two axes are
        ``(displacement component, gradient direction)``.
    :param boxsize: simulation box size.
    :return: :class:`jax.experimental.sparse.BCOO` of shape ``(N**3, N**3)``.
        The matrix is symmetric positive semidefinite with a single zero mode
        (the constant vector).
    """
    N = G.shape[0]
    assert G.shape == (N, N, N, 3, 3), f"G has shape {G.shape}, expected (N,N,N,3,3)"

    G_flat = G.reshape(N**3, 3, 3)
    conn = kuhn_connectivity(N)
    lag_edges = lagrangian_edge_vectors(N, boxsize)

    K_e, _ = _element_contributions(G_flat, conn, lag_edges)

    # Build (row, col) index pairs for each tet: outer product of conn entries.
    rows = jnp.repeat(conn, 4, axis=1).reshape(-1, 4, 4)         # (T, 4, 4)
    cols = jnp.tile(conn[:, None, :], (1, 4, 1))                 # (T, 4, 4)

    indices = jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)
    values = K_e.reshape(-1)

    return BCOO((values, indices), shape=(N**3, N**3))


def assemble_source(G: Array, boxsize: float) -> Array:
    """Assemble the lumped-mass RHS ``f_i = Σ_{tets∋i} δ_i · |V_tet|/4``.

    :param G: deformation gradient at vertices, shape ``(N, N, N, 3, 3)``.
    :param boxsize: box size.
    :return: vector of length ``N**3`` with zero mean.
    """
    N = G.shape[0]
    assert G.shape == (N, N, N, 3, 3)

    G_flat = G.reshape(N**3, 3, 3)
    conn = kuhn_connectivity(N)
    lag_edges = lagrangian_edge_vectors(N, boxsize)

    _, V_abs = _element_contributions(G_flat, conn, lag_edges)

    J_vert = jnp.linalg.det(jnp.eye(3) + G_flat)         # (N**3,)
    delta = 1.0 / J_vert - 1.0

    # Each tet contributes |V|/4 * δ to each of its 4 vertices.
    delta_per_tet_vertex = delta[conn] * (V_abs[:, None] / 4.0)  # (T, 4)

    f = jax.ops.segment_sum(
        delta_per_tet_vertex.reshape(-1),
        conn.reshape(-1),
        num_segments=N**3,
    )
    return f - f.mean()


def assemble_stiffness_stencil_from_edges(edges_def: Array, conn: Array,
                                          N: int) -> FEMStencil:
    """Build a :class:`FEMStencil` directly from deformed tet edges.

    Used by the N-body path where vertex positions (rather than a vertex-wise
    deformation gradient ``G``) are the state variable. The same
    :func:`_element_contributions_from_edges` helper backs both paths so
    element-stiffness math stays in one place.

    :param edges_def: ``(T, 3, 3)`` deformed edge vectors (``v_{k+1} − v_0``).
    :param conn: ``(T, 4)`` connectivity table for N^3 periodic grid.
    :param N: grid resolution per dimension.
    """
    K_e, _, _ = _element_contributions_from_edges(edges_def)
    return FEMStencil(K_e=K_e, conn=conn, N=N)


def assemble_source_tet(delta_tet: Array, V_abs: Array, conn: Array,
                        N: int) -> Array:
    """Lumped-mass RHS from tet-wise density contrast.

    For a piecewise-uniform density (constant per tet), the P1-FEM source
    at vertex i is ``f_i = Σ_{tets ∋ i} δ_tet · |V_tet|/4``. This is the
    physically correct discretisation when mass is carried by Lagrangian
    tet elements (our N-body case), as opposed to :func:`assemble_source`
    which evaluates a vertex-wise δ (cascade case, not mass-conserving).

    :param delta_tet: ``(T,)`` overdensity ``ρ_tet/ρ̄ − 1`` per tet.
    :param V_abs: ``(T,)`` |signed tet volumes| on the current deformed mesh.
    :param conn: ``(T, 4)`` connectivity.
    :param N: grid resolution.
    :return: ``(N**3,)`` zero-mean RHS vector.
    """
    delta_per_tet_vertex = jnp.broadcast_to(
        (delta_tet * V_abs / 4.0)[:, None], (conn.shape[0], 4)
    )
    f = jax.ops.segment_sum(
        delta_per_tet_vertex.reshape(-1),
        conn.reshape(-1),
        num_segments=N**3,
    )
    return f - f.mean()
