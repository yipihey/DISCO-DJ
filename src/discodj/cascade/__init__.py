"""FEM-based cascade solver for cosmological structure formation.

The cascade replaces the nLPT series with a loop of free Gaussian increments
each corrected by a Poisson solve on the deformed Kuhn mesh. Measures the
volume-PDF statistics exactly on the discretization for comparison with the
renormalization-group flow theory.
"""

from .fem_cascade import fem_cascade, one_lpt_increment, pk_table_from_discodj
from .fem_cascade_v2 import fem_cascade_v2, CascadeV2Result
from .fem_cascade_v3 import fem_cascade_v3, fem_cascade_v3_batch, CascadeV3Result
from .fem_cascade_v4 import fem_cascade_v4, CascadeV4Result
from .fem_cascade_v5 import fem_cascade_v5, CascadeV5Result

__all__ = [
    "fem_cascade", "fem_cascade_v2", "fem_cascade_v3", "fem_cascade_v3_batch",
    "fem_cascade_v4", "fem_cascade_v5",
    "CascadeV2Result", "CascadeV3Result", "CascadeV4Result", "CascadeV5Result",
    "one_lpt_increment", "pk_table_from_discodj",
]
