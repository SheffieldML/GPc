"""Light Python façade over GPc (CIP-001D / REQ-000E).

This package is an ease-of-use boundary for GPy users. Modelling stays in the
C++ core; NumPy arrays cross the process boundary only.

GPy-shaped entry points::

    import gpc
    k = gpc.kern.RBF(1) + gpc.kern.Bias(1) + gpc.kern.White(1)
    m = gpc.models.GPRegression(X, Y, kernel=k)

String-kernel constructors remain available for the default signal+bias+white
compound (``gpc.GPRegression(X, Y)`` / ``kernel='rbf'``).
"""

from __future__ import annotations

from . import kern, models
from ._gpc import (
    GpcError,
    Matrix,
    matrix_from_numpy,
    matrix_to_numpy,
)
from .models import GPRegression, SparseGPRegression

__all__ = [
    "GPRegression",
    "SparseGPRegression",
    "Matrix",
    "GpcError",
    "matrix_from_numpy",
    "matrix_to_numpy",
    "kern",
    "models",
    "__version__",
]

__version__ = "0.1.0a1"
