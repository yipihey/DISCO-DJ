"""Fourier-space Hessian of the gravitational potential.

Matches the kernel conventions used in :mod:`discodj.core.kernels` (specifically
``gradient_kernel``), so the result composes cleanly with
``DiscoDJ.evaluate_jacobian_from_psi``. We reuse ``gradient_kernel`` directly
rather than rolling our own so a convention change upstream propagates here.
"""

import jax.numpy as jnp
from jax import Array

from ..core.grids import get_fourier_grid
from ..core.kernels import gradient_kernel


__all__ = ["fourier_hessian"]


def fourier_hessian(phi: Array, boxsize: float) -> Array:
    """Compute ``δG_ij = -∂_i ∂_j φ`` at each grid point via FFT.

    The sign convention follows the Zel'dovich-style correction: in overdense
    regions (``φ < 0``), the trace of ``δG`` should be negative, enhancing
    collapse.

    :param phi: potential on the regular N^3 Lagrangian grid, shape
        ``(N, N, N)`` or flattened ``(N**3,)``.
    :param boxsize: simulation box size.
    :return: ``δG`` of shape ``(N, N, N, 3, 3)`` matching DiscoDJ's
        ``(displacement, gradient)`` convention.
    """
    if phi.ndim == 1:
        N = round(phi.size ** (1 / 3))
        assert N * N * N == phi.size, f"{phi.size} is not a perfect cube"
        phi = phi.reshape(N, N, N)
    N = phi.shape[0]
    assert phi.shape == (N, N, N)

    k_dict = get_fourier_grid(
        [N, N, N], boxsize=boxsize, sparse_k_vecs=True, full=False,
        dtype_num=32, relative=False, with_jax=True,
    )
    k_vecs = k_dict["k_vecs"]
    grads = [gradient_kernel(k_vecs, d, order=0, with_jax=True) for d in range(3)]

    phi_hat = jnp.fft.rfftn(phi)
    hess = jnp.zeros((N, N, N, 3, 3), dtype=phi.dtype)
    for i in range(3):
        for j in range(3):
            # −∂_i ∂_j φ = −(i k_i)(i k_j) φ̂ = k_i k_j φ̂ ; but we want the
            # Hessian contribution to G with sign opposite to the acceleration,
            # so δG_ij = −(i k_i)(i k_j) φ̂ · sign. See sign discussion in
            # fem_cascade.py.
            kern = grads[i] * grads[j]  # = (ik_i)(ik_j) = −k_i k_j
            hess_ij = jnp.fft.irfftn(phi_hat * kern, s=(N, N, N))
            hess = hess.at[..., i, j].set(hess_ij)
    return hess
