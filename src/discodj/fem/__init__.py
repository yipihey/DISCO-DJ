"""Finite-element (Kuhn tessellation) Poisson solver on the deformed Lagrangian mesh.

Used by :mod:`discodj.cascade.fem_cascade` as an alternative to the nLPT series:
each cascade step solves Poisson on the deformed mesh for an exact discrete
gravitational correction.
"""

from .kuhn import kuhn_connectivity, lagrangian_edge_vectors
from .assembly import (
    FEMStencil,
    assemble_stiffness,
    assemble_stiffness_stencil,
    assemble_stiffness_stencil_from_edges,
    assemble_source,
    assemble_source_tet,
    stencil_diag,
    stencil_matvec,
    _element_contributions_from_edges,
)
from .solve import solve_poisson
from .hessian import fourier_hessian

__all__ = [
    "kuhn_connectivity",
    "lagrangian_edge_vectors",
    "assemble_stiffness",
    "assemble_stiffness_stencil",
    "assemble_stiffness_stencil_from_edges",
    "FEMStencil",
    "stencil_matvec",
    "stencil_diag",
    "assemble_source",
    "assemble_source_tet",
    "solve_poisson",
    "fourier_hessian",
]
