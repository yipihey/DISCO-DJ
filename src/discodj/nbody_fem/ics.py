"""Bridge: pull (x, p) initial conditions straight out of a DiscoDJ object.

Rationale (see Phase 3 plan): DISCO-DJ's ``with_ics()`` uses a specific
white-noise ordering and normalisation. To compare the FEM N-body against
``DiscoDJ.run_nbody(method="pm")`` at matched realisations, we must reuse
the *same* ``DiscoDJ`` object's ψ and dψ/dD — not re-draw the GRF ourselves.

DISCO-DJ's momentum convention (verified against
``nbody/steppers/dkd_pi_integrator.py``): ``p = dx/dD`` where D is the
normalised growth factor. For 1LPT this is just ``dψ/dD|_{a_ini}`` which
equals the normalised ψ field; for higher-order LPT it's a non-trivial
mixture of LPT modes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .geometry import grid_positions


__all__ = ["ics_from_discodj"]


def ics_from_discodj(dj, a_ini: float, n_order: int = 1,
                     exact_growth: bool = False) -> tuple[Array, Array]:
    """Extract ``(x0, p0)`` from a ``DiscoDJ`` object at scale factor ``a_ini``.

    Assumes the caller has already called ``with_timetables``, ``with_linear_ps``,
    ``with_ics``, and ``with_lpt(n_order=n_order)``.

    :param dj: a configured :class:`discodj.disco_dj.DiscoDJ` instance.
    :param a_ini: initial scale factor.
    :param n_order: LPT order to evaluate; 1 = Zel'dovich. Must not exceed
        the order used for ``with_lpt``.
    :param exact_growth: pass-through to the LPT evaluator.
    :return: ``(x0, p0)`` each of shape ``(N**3, 3)`` in Mpc/h. ``p0`` is
        ``dx/dD|_{a_ini}``.
    """
    psi_ini = dj.evaluate_lpt_psi_at_a(a_ini, n_order=n_order,
                                       exact_growth=exact_growth)
    pi_ini = dj._evaluate_lpt_property_at_a(
        a=a_ini, n_order=n_order, include_psi_0=False,
        D_derivative=True, exact_growth=exact_growth,
    )

    N = dj.res
    L = dj.boxsize
    q = grid_positions(N, L)

    psi_flat = dj.ensure_flat_shape(psi_ini)
    pi_flat = dj.ensure_flat_shape(pi_ini)

    # Keep positions *unwrapped* — so x - q gives the true displacement even
    # for particles whose ZA displacement crosses the box boundary. The
    # FEM solver's Lagrangian-aware unwrapping in ``positions_to_edges``
    # expects this convention.
    x0 = (q + psi_flat).astype(jnp.float32)
    p0 = pi_flat.astype(jnp.float32)
    return x0, p0
