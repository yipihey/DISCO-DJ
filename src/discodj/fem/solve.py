"""Sparse Poisson solver wrapper for the deformed-mesh FEM stiffness matrix.

Two preconditioners are available:

- ``"jacobi"`` — diagonal of ``K``. Cheap, but for a 3D Poisson system CG
  converges in O(N) iterations which is 100s at N=64.
- ``"fft"`` (default) — the 1LPT inverse Laplacian ``K_0⁻¹ ≈ h³/|k|²`` using
  DISCO-DJ's ``inv_laplace_kernel``. On the deformed mesh ``K ≈ K_0 + O(G)``
  so this is near-perfect and drops CG to O(10) iterations.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

from ..core.grids import get_fourier_grid
from ..core.kernels import inv_laplace_kernel
from .assembly import FEMStencil, stencil_diag

__all__ = ["solve_poisson"]


def _diag_from_bcoo(K: BCOO) -> Array:
    """Extract the diagonal of a (potentially duplicated-index) BCOO matrix."""
    rows, cols = K.indices[:, 0], K.indices[:, 1]
    n = K.shape[0]
    diag_mask = (rows == cols).astype(K.data.dtype)
    diag = jax.ops.segment_sum(K.data * diag_mask, rows, num_segments=n)
    return diag


def _fft_laplacian_preconditioner(N: int, boxsize: float):
    """Build the FFT-based inverse-Laplacian preconditioner.

    On the undeformed mesh, the FEM stiffness ``K_0`` satisfies
    ``K_0 φ ≈ h³ · (-∇²) φ`` to leading order in grid spacing. So
    ``K_0⁻¹ y ≈ FFT⁻¹((-inv_laplace_kernel(k)) / h³ · FFT(y))``
    with the zero mode dropped (consistent with the projection onto the
    mean-zero subspace that the solver already enforces).

    Returns a linear function of shape-``(N³,)`` vectors.
    """
    h = boxsize / N
    k_dict = get_fourier_grid(
        [N, N, N], boxsize=boxsize, sparse_k_vecs=True, full=False,
        dtype_num=32, relative=False, with_jax=True,
    )
    k_vecs = k_dict["k_vecs"]
    # inv_laplace_kernel returns −1/|k|² with DC set to 0.
    invlap = inv_laplace_kernel(k_vecs, order=0, with_jax=True)
    inv_k2_over_h3 = (-invlap) / (h**3)  # i.e. 1/(h³ |k|²), DC = 0

    def M_inv(y: Array) -> Array:
        y3 = y.reshape(N, N, N)
        y_hat = jnp.fft.rfftn(y3)
        phi = jnp.fft.irfftn(y_hat * inv_k2_over_h3, s=(N, N, N))
        return phi.ravel()

    return M_inv


def solve_poisson(
    K: BCOO | FEMStencil,
    f: Array,
    tol: float = 1e-7,
    maxiter: int = 10,
    x0: Array | None = None,
    preconditioner: str = "fft",
    boxsize: float = 1.0,
) -> Array:
    """Solve ``K φ = f`` with preconditioned CG and zero-mean ``φ``.

    The discrete Laplacian on a periodic domain has a one-dimensional null
    space (the constant vector). We regularize by projecting both the RHS
    and the iterate onto the zero-mean subspace: ``f -= mean(f)`` (caller
    should have done this, but we do it again as a safety net) and
    ``φ -= mean(φ)`` at the end.

    :param K: FEM stiffness matrix as BCOO of shape ``(N**3, N**3)``.
    :param f: RHS vector, shape ``(N**3,)``. Must already have zero mean
        (``assemble_source`` subtracts the mean).
    :param tol: relative residual tolerance for CG.
    :param maxiter: maximum CG iterations.
    :param x0: optional warm start for CG.
    :param preconditioner: ``"fft"`` (DISCO-DJ's inverse Laplacian, default)
        or ``"jacobi"`` (diagonal of K). The FFT option drops CG iteration
        count by ~20x at N≥32 at the cost of two FFTs per iteration.
    :param boxsize: simulation box size (needed for the FFT preconditioner
        to know the grid spacing).
    :return: potential vector ``φ`` of shape ``(N**3,)`` with zero mean.
    """
    if isinstance(K, FEMStencil):
        N = K.N
        n = N**3
    else:
        n = K.shape[0]
        N = round(n ** (1 / 3))
        assert N**3 == n, f"{n} is not a perfect cube"

    if preconditioner == "fft":
        M = _fft_laplacian_preconditioner(N, boxsize)
    elif preconditioner == "jacobi":
        diag = stencil_diag(K) if isinstance(K, FEMStencil) else _diag_from_bcoo(K)
        safe_diag = jnp.where(jnp.abs(diag) > 1e-30, diag, 1.0)
        inv_diag = 1.0 / safe_diag
        M = lambda y: inv_diag * y  # noqa: E731
    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

    f_zm = f - f.mean()
    mv = lambda x: K @ x  # noqa: E731

    phi, _ = jax.scipy.sparse.linalg.cg(
        mv, f_zm, x0=x0, tol=tol, maxiter=maxiter, M=M
    )
    return phi - phi.mean()
