"""GPy-like model constructors with paramz-style display (CIP-001D / CIP-001E)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import kern as kern_mod
from ._display import OptimiserTableProgress, format_model_description
from ._gpc import BayesianGPLVM as _BayesianGPLVM
from ._gpc import GPClassification as _GPClassification
from ._gpc import GPLVM as _GPLVM
from ._gpc import GPRegression as _GPRegression
from ._gpc import MRD as _MRD
from ._gpc import OptimiserEvent
from ._gpc import SparseGPRegression as _SparseGPRegression


def _build_callback(
    model: Any,
    *,
    messages: bool,
    callback: Callable[[OptimiserEvent], Any] | None,
    display_name: str,
    max_iters: int,
) -> Callable[[OptimiserEvent], Any] | None:
    observers: list[Callable[[OptimiserEvent], Any]] = []
    if messages:
        observers.append(
            OptimiserTableProgress(
                model, name=display_name, max_iters=max_iters
            )
        )
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

    def _repr_html_(self) -> str:
        from ._display import format_model_description_html

        return format_model_description_html(
            self.describe(),
            name="GP regression",
            objective=self.objective(),
        )

    def optimize(
        self,
        iters: int = 100,
        messages: bool = False,
        callback: Callable[[OptimiserEvent], Any] | None = None,
    ) -> None:
        """Run the C++ optimiser.

        ``messages=True`` shows a paramz-style live table: ``ipywidgets`` in
        notebooks when available, else a single ``display_id`` cell, else ANSI
        rewrite in a plain terminal. It does **not** stream C++ ``Iteration:``
        / gradient-check dumps.
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
                max_iters=iters,
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
        inducing_fixed: bool = False,
        verbosity: int = 0,
        inducing: str = "points",
        window_width: Any = -1.0,
        arccos_order: int = 0,
        learn_window_width: bool = False,
        learn_window_widths: bool = False,
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
            learn_window_width=learn_window_width,
            learn_window_widths=learn_window_widths,
        )

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="Sparse GP regression",
            objective=self.objective(),
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()

    def _repr_html_(self) -> str:
        from ._display import format_model_description_html

        return format_model_description_html(
            self.describe(),
            name="Sparse GP regression",
            objective=self.objective(),
        )

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
                max_iters=iters,
            ),
        )


class GPClassification(_GPClassification):
    """FTC Bernoulli–probit classification (EP or Laplace; CIP-0033 / CIP-0036).

    Reflects GPy ``GPClassification`` for the dense FTC path:

    - ``inference='ep'`` (default) or ``'laplace'``
    - ``predict`` / ``predict_noiseless`` → latent ``f_*`` mean/variance
    - ``predict_proba`` / ``predictive_probabilities`` → ``P(y=1)``
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: Any = "rbf",
        inference: str = "ep",
        verbosity: int = 0,
    ) -> None:
        spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(
            X, Y, kernel=spec, inference=inference, verbosity=verbosity
        )

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name=f"GP classification ({self.inference})",
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
        """Optimise kernel (and bias) hyperparameters; EP/Laplace inference fixed."""
        _GPClassification.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name=f"GP classification ({self.inference})",
                max_iters=iters,
            ),
        )


class GPLVM(_GPLVM):
    """MAP Gaussian process latent variable model (``CGplvm``).

    GPy-shaped constructor: observations ``Y`` (N×D) and latent ``input_dim``.
    Default kernel matches the native ``./gplvm`` CLI (RBF + Bias + White).
    """

    def __init__(
        self,
        Y: np.ndarray,
        input_dim: int = 2,
        kernel: Any = "rbf",
        verbosity: int = 0,
    ) -> None:
        spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(Y, input_dim=input_dim, kernel=spec, verbosity=verbosity)

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="GPLVM",
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
        """Optimise latent positions and kernel hyperparameters."""
        _GPLVM.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name="GPLVM",
                max_iters=iters,
            ),
        )


class BayesianGPLVM(_BayesianGPLVM):
    """Collapsed Bayesian GPLVM with psi statistics (``CBayesGplvm``; CIP-0017).

    GPy-shaped constructor: ``Y``, ``input_dim``, ``num_inducing``. Default
    kernel is RBF-ARD when ``ard=True`` (GPy ``BayesianGPLVM`` parity).
    """

    def __init__(
        self,
        Y: np.ndarray,
        input_dim: int = 2,
        num_inducing: int = 10,
        kernel: Any = "rbf",
        ard: bool = True,
        verbosity: int = 0,
    ) -> None:
        spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(
            Y,
            input_dim=input_dim,
            num_inducing=num_inducing,
            kernel=spec,
            ard=ard,
            verbosity=verbosity,
        )

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="Bayesian GPLVM",
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
        """Optimise variational ``q(X)``, inducing inputs, kernel, and β."""
        _BayesianGPLVM.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name="Bayesian GPLVM",
                max_iters=iters,
            ),
        )


class MRD(_MRD):
    """Manifold Relevance Determination (``CMrd``; CIP-0018).

    GPy-shaped constructor: ``Ylist`` (aligned views), ``input_dim``,
    ``num_inducing``. Default per-view kernel is bare RBF-ARD so
    :meth:`factorize_space` works.
    """

    def __init__(
        self,
        Ylist: Any,
        input_dim: int = 2,
        num_inducing: int = 10,
        kernel: Any = None,
        ard: bool = True,
        verbosity: int = 0,
    ) -> None:
        if kernel is None:
            spec: Any = None
        elif isinstance(kernel, (list, tuple)) and kernel and not isinstance(
            kernel[0], dict
        ):
            # Per-view specs (strings / Kern / parts lists).
            spec = [
                None
                if k is None
                else kern_mod.normalize_kernel_arg(k)
                for k in kernel
            ]
        else:
            spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(
            Ylist,
            input_dim=input_dim,
            num_inducing=num_inducing,
            kernel=spec,
            ard=ard,
            verbosity=verbosity,
        )

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="MRD",
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
        """Optimise shared latents and per-view kernels."""
        _MRD.optimize(
            self,
            iters=iters,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name="MRD",
                max_iters=iters,
            ),
        )


__all__ = [
    "GPRegression",
    "SparseGPRegression",
    "GPClassification",
    "GPLVM",
    "BayesianGPLVM",
    "MRD",
]
