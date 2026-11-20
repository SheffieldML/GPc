"""GPy-like model constructors for the `gpc` façade (CIP-001D)."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import kern as kern_mod
from ._gpc import GPRegression as _GPRegression
from ._gpc import SparseGPRegression as _SparseGPRegression


def GPRegression(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Any = "rbf",
    verbosity: int = 0,
    arccos_order: int = 0,
):
    """FTC Gaussian regression.

    Parameters
    ----------
    X, Y
        Training inputs/outputs (NumPy arrays).
    kernel
        String name (``'rbf'``, ``'eq'`` / ``'ExpQuad'``, ``'mlp'``, ``'arccos'``)
        for the default signal+bias+white compound, or a ``gpc.kern`` object /
        compound (built exactly as specified — add ``White`` / ``Bias`` yourself
        for GPy-style recipes).
    """
    spec = kern_mod.normalize_kernel_arg(kernel)
    if isinstance(spec, str):
        return _GPRegression(
            X, Y, kernel=spec, verbosity=verbosity, arccos_order=arccos_order
        )
    return _GPRegression(
        X, Y, kernel=spec, verbosity=verbosity, arccos_order=arccos_order
    )


def SparseGPRegression(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    kernel: Any = "rbf",
    approx: str = "dtcvar",
    inducing_fixed: bool = True,
    verbosity: int = 0,
    inducing: str = "points",
    window_width: float = -1.0,
    arccos_order: int = 0,
):
    """Sparse Gaussian regression (DTCVAR / FITC / DTC).

    ``kernel`` accepts the same string names or ``gpc.kern`` compounds as
    :func:`GPRegression`. Custom compounds are supported for ``inducing='points'``;
    ``inducing='weightspace'`` still requires a string ``mlp`` / ``arccos`` kernel.
    """
    spec = kern_mod.normalize_kernel_arg(kernel)
    return _SparseGPRegression(
        X,
        Y,
        Z,
        approx=approx,
        inducing_fixed=inducing_fixed,
        verbosity=verbosity,
        kernel=spec,
        inducing=inducing,
        window_width=window_width,
        arccos_order=arccos_order,
    )


__all__ = ["GPRegression", "SparseGPRegression"]
