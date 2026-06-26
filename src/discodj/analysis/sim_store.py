"""Loader + common derived quantities for the stored PM sim suite."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..core.grids import get_fourier_grid, get_lagrangian_grid_vectors
from ..core.kernels import gradient_kernel


__all__ = [
    "SimSnapshot",
    "load_sim",
    "list_sims",
    "displacement_field",
    "deformation_gradient_jacobian",
]


@dataclass
class SimSnapshot:
    """One snapshot of a PM sim. ``x`` is ``(N**3, 3)`` in Mpc/h."""

    x: np.ndarray
    v: np.ndarray
    a: float
    N: int
    L: float
    meta: dict

    @property
    def h(self) -> float:
        return self.L / self.N

    @property
    def q(self) -> np.ndarray:
        """Lagrangian grid, same C-order as DISCO-DJ stores particles."""
        h = self.h
        ix, iy, iz = np.meshgrid(
            np.arange(self.N), np.arange(self.N), np.arange(self.N), indexing="ij"
        )
        return np.stack([ix.ravel(), iy.ravel(), iz.ravel()], -1).astype(np.float32) * h


def list_sims(root: str = "sims") -> list[str]:
    """Return tags (stem of the .npz) for available sims in ``root``."""
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(root, "N*.npz"))
    )


def load_sim(N: int, cutoff_mode: str, a: float | None = None,
             root: str = "sims") -> SimSnapshot | list[SimSnapshot]:
    """Load a stored sim. If a low-a companion file (``*_lowa.npz``)
    exists, its snapshots are merged in transparently."""
    paths = []
    for stem in (f"N{N}_{cutoff_mode}", f"N{N}_{cutoff_mode}_lowa"):
        p = os.path.join(root, f"{stem}.npz")
        if os.path.exists(p):
            paths.append(p)
    if not paths:
        raise FileNotFoundError(
            f"No sims found matching N{N}_{cutoff_mode} under {root}/")

    snaps = []
    meta_ref = None
    for path in paths:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        meta_ref = meta_ref or meta
        xs, vs, as_ = data["x"], data["v"], data["a"]
        for i in range(len(as_)):
            snaps.append(SimSnapshot(x=xs[i], v=vs[i], a=float(as_[i]),
                                     N=meta["N"], L=meta["L"], meta=meta))
    snaps.sort(key=lambda s: s.a)

    if a is None:
        return snaps
    idx = int(np.argmin(np.abs(np.asarray([s.a for s in snaps]) - a)))
    return snaps[idx]


def displacement_field(snap: SimSnapshot, as_mesh: bool = True) -> np.ndarray:
    """Return the periodic-min-image displacement ``ψ = x − q``.

    :param as_mesh: if True, reshape to ``(N, N, N, 3)``.
    """
    psi = snap.x - snap.q
    psi = psi - snap.L * np.round(psi / snap.L)
    if as_mesh:
        return psi.reshape(snap.N, snap.N, snap.N, 3)
    return psi


def deformation_gradient_jacobian(snap: SimSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(G, J)`` via Fourier-space differentiation of ψ.

    Returns:
        G: shape ``(N, N, N, 3, 3)`` with ``G_{di} = ∂ψ_d / ∂q_i``.
        J: shape ``(N, N, N)`` Jacobian determinant of ``I + G``.
    """
    psi_mesh = jnp.asarray(displacement_field(snap, as_mesh=True))
    N = snap.N
    L = snap.L
    kd = get_fourier_grid([N] * 3, boxsize=L, sparse_k_vecs=True, full=False,
                          dtype_num=32, with_jax=True)
    k_vecs = kd["k_vecs"]
    grads = [gradient_kernel(k_vecs, d, order=0, with_jax=True) for d in range(3)]

    G = np.zeros((N, N, N, 3, 3), dtype=np.float32)
    for d in range(3):
        psi_d_hat = jnp.fft.rfftn(psi_mesh[..., d])
        for i in range(3):
            gd_i = jnp.fft.irfftn(grads[i] * psi_d_hat, s=(N, N, N))
            G[..., d, i] = np.asarray(gd_i, dtype=np.float32)

    F = np.eye(3, dtype=np.float32)[None, None, None, :, :] + G
    J = np.linalg.det(F)
    return G, J
