"""Phase-space-sheet (PSS) tetrahedron volumes from PM snapshots.

The particles trace a 3D sheet in 6D phase space. Connect each Lagrangian
cubic cell into 6 Kuhn tetrahedra; their Eulerian volumes give a direct,
*signed* volume distribution at a well-defined Lagrangian mass scale —
no Poisson shot noise, no density estimator.

The core volume computation :func:`_kuhn_tet_volumes_jit` is pure JAX and
jit-compiled, so it runs on GPU/TPU when available and stays
differentiable: gradients of any volume-derived statistic w.r.t. the
particle positions flow through cleanly via ``jax.grad``/``jax.jvp``.

Multi-scale is controlled by ``stride``: the Kuhn cube at origin
``(i, j, k)`` has corners at ``(i + s·di, j + s·dj, k + s·dk)`` for
``(di, dj, dk) ∈ {0, 1}³``. By default the sampler is **overlapping** —
cubes are formed at every Lagrangian origin (N³ per stride), not just
every s-th origin. Neighbours share vertices so samples are correlated,
but the factor-of-s³ boost in cube count crushes the κ_3 noise floor.

Pass ``overlapping=False`` for strict-subsampling behaviour (``(N/s)³``).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .sim_store import displacement_field, SimSnapshot


__all__ = ["kuhn_tet_volumes", "kuhn_cube_volumes",
           "kuhn_tet_volumes_multi_stride", "parent_child_cube_pairs",
           "kuhn_cube_volumes_from_positions",
           "kuhn_tet_deformation_tensor_from_positions",
           "strain_invariants_from_F",
           "KUHN_VERTEX_OFFSETS"]


# Six Kuhn tets per unit cube, consistently oriented (det = +1) — same
# convention as ``discodj.fem.kuhn._KUHN_TETS``. Corner index
# ``k = 4·cz + 2·cy + cx`` with ``(cx, cy, cz) ∈ {0, 1}³``.
_CORNER = [(k & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
_KUHN_TET_IDX = [
    [0, 1, 3, 7],
    [0, 1, 7, 5],
    [0, 2, 7, 3],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 7, 6],
]
KUHN_VERTEX_OFFSETS = tuple(
    tuple(_CORNER[i] for i in tet) for tet in _KUHN_TET_IDX)

_KUHN_OFFSETS_ARRAY = np.asarray(KUHN_VERTEX_OFFSETS, dtype=np.int32)

# Lagrangian edge matrices: E_lag[t, :, k] = edge k (k=0,1,2) of tet t
# stacked as columns of a 3x3 matrix. Each edge is the Lagrangian vertex
# offset relative to vertex 0, so E_lag is integer-valued with det=+1 for
# all six tets. Used to recover the deformation gradient
#   F_t = E_eul_t · E_lag_t^{-1}
# from measured Eulerian edges.
_KUHN_LAG_EDGES = np.zeros((6, 3, 3), dtype=np.float32)
for _t, _tet in enumerate(_KUHN_TET_IDX):
    for _k in (1, 2, 3):
        edge = np.asarray(_CORNER[_tet[_k]], np.float32) - \
                np.asarray(_CORNER[_tet[0]], np.float32)
        _KUHN_LAG_EDGES[_t, :, _k - 1] = edge
_KUHN_LAG_EDGES_INV = np.linalg.inv(_KUHN_LAG_EDGES).astype(np.float32)


# ---------------------------------------------------------------------
# Core JAX kernels
# ---------------------------------------------------------------------


@partial(jax.jit, static_argnames=("stride", "overlapping"))
def _kuhn_tet_volumes_jit(x: jnp.ndarray, L: float,
                           stride: int, overlapping: bool) -> jnp.ndarray:
    """JAX kernel returning ``(6, Nc, Nc, Nc)`` signed tet volumes.

    ``x`` has shape ``(N, N, N, 3)`` and holds the *unwrapped* Eulerian
    positions, matched index-for-index to the Lagrangian grid. ``Nc`` is
    ``N`` when ``overlapping`` is ``True`` and ``N // stride`` otherwise.
    """
    N = x.shape[0]
    # Strict-subsample path: extract every stride-th Lagrangian point first,
    # then always shift by 1 on the coarse grid.
    if not overlapping:
        x_use = x[::stride, ::stride, ::stride]
        shift_unit = 1
    else:
        x_use = x
        shift_unit = stride

    vols_per_tet = []
    for t_idx in range(6):
        off = _KUHN_OFFSETS_ARRAY[t_idx]            # (4, 3)
        pts = []
        for v_idx in range(4):
            dx = int(off[v_idx, 0])
            dy = int(off[v_idx, 1])
            dz = int(off[v_idx, 2])
            if dx == 0 and dy == 0 and dz == 0:
                shifted = x_use
            else:
                # Shift then patch the wrap-around edges with ±L so edges stay short.
                s = shift_unit
                shifted = jnp.roll(x_use, shift=(-s * dx, -s * dy, -s * dz),
                                    axis=(0, 1, 2))
                if dx == 1:
                    shifted = shifted.at[-s:, :, :, 0].add(L)
                if dy == 1:
                    shifted = shifted.at[:, -s:, :, 1].add(L)
                if dz == 1:
                    shifted = shifted.at[:, :, -s:, 2].add(L)
            pts.append(shifted)
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[0]
        e3 = pts[3] - pts[0]
        cross = jnp.cross(e1, e2)
        vols_per_tet.append(jnp.einsum("...i,...i->...", cross, e3) / 6.0)
    return jnp.stack(vols_per_tet, axis=0)


@partial(jax.jit, static_argnames=("stride", "overlapping"))
def _kuhn_cube_volumes_jit(x: jnp.ndarray, L: float,
                            stride: int, overlapping: bool) -> jnp.ndarray:
    return _kuhn_tet_volumes_jit(x, L, stride, overlapping).sum(axis=0)


@partial(jax.jit, static_argnames=("stride", "overlapping"))
def _kuhn_tet_F_jit(x: jnp.ndarray, L: float,
                     stride: int, overlapping: bool) -> jnp.ndarray:
    """JAX kernel returning ``(6, Nc, Nc, Nc, 3, 3)`` deformation gradients.

    ``F[t, i, j, k] = ∂x/∂q`` for tet ``t`` anchored at Lagrangian cell
    ``(i, j, k)``. The Jacobian ``J = det(F)`` recovers the 6-tet signed
    volumes up to the prefactor ``d_lag³ = (s·L/N)³``.
    """
    N = x.shape[0]
    if not overlapping:
        x_use = x[::stride, ::stride, ::stride]
        shift_unit = 1
    else:
        x_use = x
        shift_unit = stride
    d_lag = (stride if overlapping else 1) * (L / (N if overlapping else x_use.shape[0]))

    # Dimensionless step per edge: the Lagrangian edge components are in
    # units of d_lag, so dividing the Eulerian edges by d_lag puts F into
    # dimensionless form.
    lag_inv = jnp.asarray(_KUHN_LAG_EDGES_INV, dtype=x.dtype)

    Fs = []
    for t_idx in range(6):
        off = _KUHN_OFFSETS_ARRAY[t_idx]
        pts = []
        for v_idx in range(4):
            dx = int(off[v_idx, 0])
            dy = int(off[v_idx, 1])
            dz = int(off[v_idx, 2])
            if dx == 0 and dy == 0 and dz == 0:
                shifted = x_use
            else:
                s = shift_unit
                shifted = jnp.roll(x_use, shift=(-s * dx, -s * dy, -s * dz),
                                    axis=(0, 1, 2))
                if dx == 1:
                    shifted = shifted.at[-s:, :, :, 0].add(L)
                if dy == 1:
                    shifted = shifted.at[:, -s:, :, 1].add(L)
                if dz == 1:
                    shifted = shifted.at[:, :, -s:, 2].add(L)
            pts.append(shifted)
        # Stack 3 Eulerian edge vectors as columns of a 3×3 matrix per cell.
        E_eul = jnp.stack([pts[1] - pts[0], pts[2] - pts[0], pts[3] - pts[0]],
                            axis=-1)                        # (Nc, Nc, Nc, 3, 3)
        # Divide by d_lag to make E_eul dimensionless, then multiply by the
        # inverse Lagrangian edge matrix for this tet type.
        F_t = (E_eul / d_lag) @ lag_inv[t_idx]
        Fs.append(F_t)
    return jnp.stack(Fs, axis=0)                            # (6, Nc, Nc, Nc, 3, 3)


def kuhn_tet_deformation_tensor_from_positions(
    x: jnp.ndarray, L: float, stride: int = 1, overlapping: bool = True
) -> jnp.ndarray:
    """Differentiable per-tet deformation gradient ``F = ∂x/∂q``.

    Returns a JAX array of shape ``(6, Nc, Nc, Nc, 3, 3)`` where entry
    ``F[t, i, j, k]`` is the dimensionless 3×3 gradient at the tet ``t``
    anchored in Lagrangian cell ``(i, j, k)``. The Jacobian
    ``J = det(F)`` matches the signed volume divided by the Lagrangian
    cube volume ``(s·L/N)³``.

    The invariants of interest — ``I_1 = tr(G)``, ``I_2, I_3``, and the
    shear moments ``tr(Σ²), tr(Σ³)`` — follow from ``F`` via
    :func:`strain_invariants_from_F`.
    """
    return _kuhn_tet_F_jit(x, float(L), int(stride), bool(overlapping))


@jax.jit
def strain_invariants_from_F(F: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Strain invariants and shear moments from the deformation gradient.

    Let ``G = F − 1`` and ``G_sym = (G + G^T)/2`` be the symmetric strain
    tensor (equal to ``G`` for potential flow / nLPT). ``Σ = G_sym −
    (I_1/3)·1`` is the traceless shear tensor. Returns:

    * ``I1 = tr(G)``, ``I2, I3`` — standard scalar invariants of ``G``.
    * ``trG2 = tr(G · G)``.
    * ``trS2 = tr(Σ²)``, ``trS3 = tr(Σ³)`` — symmetric-shear invariants,
      guaranteed ≥ 0 for ``trS2`` (built from the symmetric part).
    * ``antisym_sq`` — Frobenius norm squared of the antisymmetric part,
      diagnostic of curl contamination (should be ~0 for potential flow).

    Applies element-wise over any leading shape; ``F`` is expected to
    have last two axes 3×3.
    """
    eye = jnp.eye(3, dtype=F.dtype)
    G = F - eye
    I1 = jnp.einsum("...ii->...", G)
    G2 = jnp.einsum("...ij,...jk->...ik", G, G)
    trG2 = jnp.einsum("...ii->...", G2)
    I2 = 0.5 * (I1 * I1 - trG2)
    I3 = jnp.linalg.det(G)
    # Symmetric/antisymmetric split.
    G_sym = 0.5 * (G + jnp.swapaxes(G, -1, -2))
    G_asym = 0.5 * (G - jnp.swapaxes(G, -1, -2))
    antisym_sq = jnp.einsum("...ij,...ij->...", G_asym, G_asym)
    iso = (I1 / 3.0)[..., None, None] * eye
    Sigma = G_sym - iso
    Sigma2 = jnp.einsum("...ij,...jk->...ik", Sigma, Sigma)
    trS2 = jnp.einsum("...ii->...", Sigma2)
    Sigma3 = jnp.einsum("...ij,...jk->...ik", Sigma2, Sigma)
    trS3 = jnp.einsum("...ii->...", Sigma3)
    return dict(I1=I1, I2=I2, I3=I3, trG2=trG2,
                trS2=trS2, trS3=trS3, antisym_sq=antisym_sq)


def kuhn_cube_volumes_from_positions(x: jnp.ndarray, L: float,
                                       stride: int = 1,
                                       overlapping: bool = True
                                       ) -> jnp.ndarray:
    """Differentiable JAX entry point.

    Feeds unwrapped Eulerian positions ``x`` of shape ``(N, N, N, 3)`` and
    returns the per-cube signed volumes as a JAX array — suitable for
    use inside a gradient chain (``jax.grad``, ``jax.jvp``, etc.). The
    ``stride`` and ``overlapping`` arguments are static.
    """
    return _kuhn_cube_volumes_jit(x, float(L), int(stride), bool(overlapping))


# ---------------------------------------------------------------------
# Helpers: build the unwrapped Eulerian grid from a ``SimSnapshot``
# ---------------------------------------------------------------------


def _eulerian_positions(snap: SimSnapshot) -> tuple[np.ndarray, float, int]:
    """Return ``(x, L, N)`` with ``x`` shape ``(N, N, N, 3)``, minimum-image
    unwrapped so the 8-corner positions on any Lagrangian stencil are
    continuous (no ±L jumps). Returned as a numpy array — cast to jax
    inside the jitted kernels.
    """
    N = snap.N
    L = snap.L
    psi = displacement_field(snap, as_mesh=True)
    h = L / N
    ix, iy, iz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                              indexing="ij")
    q = np.stack([ix * h, iy * h, iz * h], axis=-1).astype(np.float32)
    x = q + psi.astype(np.float32)
    return x, L, N


# ---------------------------------------------------------------------
# Numpy-friendly public API
# ---------------------------------------------------------------------


def kuhn_tet_volumes(snap: SimSnapshot, stride: int = 1,
                       overlapping: bool = True) -> np.ndarray:
    """Signed Kuhn-tet volumes at ``stride``.

    - ``overlapping=True`` (default): ``6·N³`` tets at every origin.
    - ``overlapping=False``: ``6·(N/s)³`` strict-subsample tets.
    """
    x, L, _ = _eulerian_positions(snap)
    vols = _kuhn_tet_volumes_jit(jnp.asarray(x), float(L),
                                   int(stride), bool(overlapping))
    return np.asarray(vols).ravel()


def kuhn_cube_volumes(snap: SimSnapshot, stride: int = 1,
                       overlapping: bool = True) -> np.ndarray:
    """Per-cube Eulerian volume ``V_cube = Σ_{t=1..6} V_tet`` at ``stride``."""
    x, L, _ = _eulerian_positions(snap)
    vols = _kuhn_cube_volumes_jit(jnp.asarray(x), float(L),
                                    int(stride), bool(overlapping))
    return np.asarray(vols)


def kuhn_tet_volumes_multi_stride(
    snap: SimSnapshot, strides=(1, 2, 4, 8, 16, 32), overlapping: bool = True,
) -> dict[int, np.ndarray]:
    """Cube volumes at several strides; returns ``{stride: V_cube_array}``.

    Jit caches per ``(stride, overlapping)`` signature, so repeated calls
    at the same settings reuse the compiled kernel. The expensive piece
    is the Eulerian-position build (once per snapshot).
    """
    x, L, _ = _eulerian_positions(snap)
    x_j = jnp.asarray(x)
    out = {}
    for s in strides:
        if (not overlapping) and snap.N % s != 0:
            continue
        vols = _kuhn_cube_volumes_jit(x_j, float(L), int(s), bool(overlapping))
        out[s] = np.asarray(vols)
    return out


# ---------------------------------------------------------------------
# Parent-child pairing
# ---------------------------------------------------------------------


def parent_child_cube_pairs(V_parent: np.ndarray, V_child: np.ndarray,
                              stride_parent: int | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Align parent cubes with their 8 stride-s/2 children.

    * Overlapping case: both arrays have shape ``(N, N, N)`` and
      ``stride_parent`` is required. Children of parent ``(i, j, k)`` are at
      ``(i + di·s/2, j + dj·s/2, k + dk·s/2)``.
    * Non-overlapping case: parent shape ``(N_p,)*3``, child shape
      ``(2·N_p,)*3``.

    Returns:
        parents:             shape ``(M,)``
        children_per_parent: shape ``(M, 8)`` in corner-index order
                              ``i = 4·dk + 2·dj + di``.
    """
    if V_parent.shape == V_child.shape:
        if stride_parent is None:
            raise ValueError(
                "stride_parent must be passed in the overlapping case")
        if stride_parent % 2 != 0 or stride_parent < 2:
            raise ValueError("stride_parent must be even and >= 2")
        s_half = stride_parent // 2
        N = V_parent.shape[0]
        children = np.empty((N, N, N, 8), dtype=V_child.dtype)
        for k_idx in range(8):
            dk = (k_idx >> 2) & 1
            dj = (k_idx >> 1) & 1
            di = k_idx & 1
            if (di, dj, dk) == (0, 0, 0):
                children[..., 0] = V_child
            else:
                children[..., k_idx] = np.roll(
                    V_child,
                    shift=(-di * s_half, -dj * s_half, -dk * s_half),
                    axis=(0, 1, 2))
        return V_parent.ravel(), children.reshape(N**3, 8)

    N_p = V_parent.shape[0]
    if V_child.shape != (2 * N_p, 2 * N_p, 2 * N_p):
        raise ValueError(
            f"non-overlapping case needs child shape 2·{N_p}; got {V_child.shape}")
    children = np.empty((N_p, N_p, N_p, 8), dtype=V_child.dtype)
    for k_idx in range(8):
        dk = (k_idx >> 2) & 1
        dj = (k_idx >> 1) & 1
        di = k_idx & 1
        children[..., k_idx] = V_child[di::2, dj::2, dk::2]
    return V_parent.ravel(), children.reshape(N_p**3, 8)
