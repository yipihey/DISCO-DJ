"""Kuhn tessellation of a periodic N^3 grid into 6 tetrahedra per cube."""

from functools import lru_cache

import numpy as onp
import jax.numpy as jnp
from jax import Array


__all__ = ["kuhn_connectivity", "lagrangian_edge_vectors"]


# Kuhn triangulation: 6 tetrahedra covering the unit cube.
# Each row lists the four corner indices (0..7) forming a tet; the cube
# corner index k = 4*cz + 2*cy + cx with (cx, cy, cz) in {0,1}^3. All six
# tets share vertex 0 (0,0,0) and vertex 7 (1,1,1) — the long diagonal —
# and each moves along axes in a different order, tiling the cube exactly.
_KUHN_TETS = onp.asarray(
    [
        [0, 1, 3, 7],
        [0, 1, 7, 5],
        [0, 2, 7, 3],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 7, 6],
    ],
    dtype=onp.int64,
)

# Corner offsets in (cx, cy, cz) order matching the index k = 4*cz+2*cy+cx.
_CORNER_OFFSETS = onp.asarray(
    [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
    dtype=onp.int64,
)


@lru_cache(maxsize=16)
def kuhn_connectivity(N: int) -> Array:
    """Build the tetrahedron -> vertex index table for a periodic N^3 grid.

    Vertex flat index is ``ix * N*N + iy * N + iz`` with periodic wrap
    (``ix mod N`` etc.). The 6 Kuhn tets are generated per cube using the
    splitting ``[[0,1,3,7], [0,1,7,5], [0,2,7,3], [0,2,6,7], [0,4,5,7],
    [0,4,7,6]]`` over the 8 corners of the cube
    ``(ix..ix+1, iy..iy+1, iz..iz+1)``.

    :param N: grid resolution per dimension.
    :return: int32 array of shape ``(6*N**3, 4)`` with global vertex indices.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")

    # Cube origin indices — shape (N^3, 3).
    ixs, iys, izs = onp.meshgrid(
        onp.arange(N), onp.arange(N), onp.arange(N), indexing="ij"
    )
    cube_origins = onp.stack([ixs.ravel(), iys.ravel(), izs.ravel()], axis=-1)

    # For each cube, compute the 8 corner vertex flat indices with periodic wrap.
    # shape (N^3, 8, 3) -> (N^3, 8)
    corners = (cube_origins[:, None, :] + _CORNER_OFFSETS[None, :, :]) % N
    vert_flat = corners[..., 0] * (N * N) + corners[..., 1] * N + corners[..., 2]

    # For each cube, pick the 4 corners for each of the 6 tets.
    # shape (N^3, 6, 4)
    tets_per_cube = vert_flat[:, _KUHN_TETS]
    # Return a numpy array — the caller converts to jax inside its own
    # trace context. Caching a jax array across jit boundaries leaks
    # device tracers (the failure we hit in cascade_v2's midpoint path).
    return tets_per_cube.reshape(-1, 4).astype(onp.int32)


def lagrangian_edge_vectors(N: int, boxsize: float) -> Array:
    """Return the Lagrangian edges ``e_k = v_k - v_0`` for each Kuhn tet type.

    On the undeformed grid the six tets per cube have fixed edge vectors (only
    depending on the cube spacing ``h = boxsize / N``). We return them as a
    ``(6, 3, 3)`` float array where the middle axis selects edge index
    ``k in {1,2,3}`` (edges from vertex 0 in the tet) and the last axis is the
    spatial component. This is what :mod:`assembly` uses before deforming.

    :param N: grid resolution per dimension.
    :param boxsize: simulation box size in comoving units.
    """
    h = boxsize / N
    offsets = _CORNER_OFFSETS.astype(onp.float64)  # (8, 3)
    # For each tet, take v1-v0, v2-v0, v3-v0 in the corner-offset space.
    edges = onp.stack(
        [offsets[_KUHN_TETS[:, k]] - offsets[_KUHN_TETS[:, 0]] for k in (1, 2, 3)],
        axis=1,
    )  # (6, 3, 3)
    return jnp.asarray(edges * h)
