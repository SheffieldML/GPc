"""Light Python façade over GPc (CIP-001D / REQ-000E).

This package is an ease-of-use boundary for GPy users. Modelling stays in the
C++ core; NumPy arrays cross the process boundary only.

GPy-shaped entry points::

    import gpc
    k = gpc.kern.RBF(1)  # CIP-0027: Bias/White opt-in; noise is likelihood.variance
    m = gpc.models.GPRegression(X, Y, kernel=k)

String-kernel constructors remain available for the CIP-0027 signal-only
default (``gpc.GPRegression(X, Y)`` / ``kernel='rbf'``; noise is
``likelihood.variance``).
"""

from __future__ import annotations

from . import kern, models
from ._display import OptimiserTableProgress, format_model_description
from ._gpc import (
    GpcError,
    Matrix,
    ModelDescription,
    OptimiserEvent,
    OptimiserPhase,
    ParamRow,
    matrix_from_numpy,
    matrix_to_numpy,
)
from .models import GPRegression, SparseGPRegression

__all__ = [
    "GPRegression",
    "SparseGPRegression",
    "Matrix",
    "GpcError",
    "OptimiserEvent",
    "OptimiserPhase",
    "ModelDescription",
    "ParamRow",
    "format_model_description",
    "OptimiserTableProgress",
    "matrix_from_numpy",
    "matrix_to_numpy",
    "kern",
    "models",
    "__version__",
]

__version__ = "0.1.0a2"
