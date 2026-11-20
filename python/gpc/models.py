"""GPy-like model constructors with paramz-style display (CIP-001D / CIP-001E)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import kern as kern_mod
from ._display import OptimiserTableProgress, format_model_description
from ._gpc import GPRegression as _GPRegression
from ._gpc import OptimiserEvent
from ._gpc import SparseGPRegression as _SparseGPRegression


def _build_callback(
    model: Any,
    *,
    messages: bool,
    callback: Callable[[OptimiserEvent], Any] | None,
    display_name: str,
) -> Callable[[OptimiserEvent], Any] | None:
    observers: list[Callable[[OptimiserEvent], Any]] = []
    if messages:
        observers.append(OptimiserTableProgress(model, name=display_name))
    if callback is not None:
        observers.append(callback)
    if not observers:
        return None

    def _fanout(event: OptimiserEvent) -> None:
        for obs in observers:
            obs(event)

    return _fanout


class GPRegression(_GPRegression):
    """FTC Gaussian regression; ``str``/``repr`` use a paramz-inspired table."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: Any = "rbf",
        verbosity: int = 0,
        arccos_order: int = 0,
    ) -> None:
        spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(
            X, Y, kernel=spec, verbosity=verbosity, arccos_order=arccos_order
        )

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="GP regression",
            objective=self.objective(),
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()

    def optimize(
        self,
        iters: int = 100,
        messages: bool = False,
        callback: Callable[[OptimiserEvent], Any] | None = None,
    ) -> None:
        """Run the C++ optimiser.

        ``messages=True`` refreshes the paramz-style parameter table in place
        (Jupyter ``display_id`` / ANSI rewrite in a plain terminal). It does
        **not** stream C++ ``Iteration:`` / gradient-check dumps.
        """
        # C++ CLI progress / verbosity stay off; Python owns messaging.
        _GPRegression.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name="GP regression",
            ),
        )


class SparseGPRegression(_SparseGPRegression):
    """Sparse Gaussian regression; ``str``/``repr`` use a paramz-inspired table."""

    def __init__(
        self,
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
    ) -> None:
        spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(
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

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="Sparse GP regression",
            objective=self.objective(),
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()

    def optimize(
        self,
        iters: int = 100,
        messages: bool = False,
        callback: Callable[[OptimiserEvent], Any] | None = None,
    ) -> None:
        """Run the C++ optimiser (same messaging rules as :class:`GPRegression`)."""
        _SparseGPRegression.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name="Sparse GP regression",
            ),
        )


__all__ = ["GPRegression", "SparseGPRegression"]
