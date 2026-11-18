"""Light Python façade over GPc (CIP-001D / REQ-000E).

This package is an ease-of-use boundary for GPy users. Modelling stays in the
C++ core; NumPy arrays cross the process boundary only.
"""

from __future__ import annotations

from ._gpc import (
    GPRegression,
    GpcError,
    Matrix,
    ModelDescription,
    OptimiserEvent,
    OptimiserPhase,
    ParamRow,
    matrix_from_numpy,
    matrix_to_numpy,
)

__all__ = [
    "GPRegression",
    "Matrix",
    "GpcError",
    "OptimiserEvent",
    "OptimiserPhase",
    "ModelDescription",
    "ParamRow",
    "matrix_from_numpy",
    "matrix_to_numpy",
    "__version__",
]

__version__ = "0.1.0a1"
