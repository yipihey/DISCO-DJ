"""Direct N-body solver with FEM Poisson gravity on the Kuhn tessellation.

Particles live at vertices of a Kuhn tessellation of the Lagrangian grid.
Each tetrahedron is a Lagrangian fluid element with fixed mass
``m_tet = ρ̄ · V_tet(t=0)``; density ``ρ_tet(t) = m_tet / V_tet(t)`` tracks
the current deformed geometry. At each leapfrog step the FEM machinery from
:mod:`discodj.fem` assembles and solves Poisson on the current deformed
mesh; the gravitational force comes from ``F = −∂U/∂q`` with
``U(q) = ½ f(q)ᵀ K(q)⁻¹ f(q)``, giving a variational (exact-symplectic)
integrator before shell crossing.
"""

from .geometry import positions_to_edges, tet_densities, grid_positions
from .energy import potential_energy, kinetic_energy
from .force import force_fn
from .integrator import kdk_step, leapfrog_run
from .run import fem_nbody, fem_nbody_cosmo, NBodyResult
from .cosmo import analytic_F_scale, build_time_table, cosmological_kdk_step
from .ics import ics_from_discodj

__all__ = [
    "positions_to_edges",
    "tet_densities",
    "grid_positions",
    "potential_energy",
    "kinetic_energy",
    "force_fn",
    "kdk_step",
    "leapfrog_run",
    "fem_nbody",
    "fem_nbody_cosmo",
    "NBodyResult",
    "analytic_F_scale",
    "build_time_table",
    "cosmological_kdk_step",
    "ics_from_discodj",
]
