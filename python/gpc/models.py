"""GPy-like model constructors with paramz-style display (CIP-001D / CIP-001E)."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import kern as kern_mod
from ._display import OptimiserTableProgress, format_model_description
from ._gpc import BayesianGPLVM as _BayesianGPLVM
from ._gpc import DeepGP as _DeepGP
from ._gpc import GPClassification as _GPClassification
from ._gpc import GPLVM as _GPLVM
from ._gpc import GPRegression as _GPRegression
from ._gpc import MRD as _MRD
from ._gpc import OptimiserEvent
from ._gpc import SparseGPRegression as _SparseGPRegression
from .util import multioutput as mo


def _contains_coregionalize(kernel: kern_mod.Kern) -> bool:
    if isinstance(kernel, kern_mod.Coregionalize):
        return True
    if isinstance(kernel, kern_mod.Add):
        return any(_contains_coregionalize(leaf) for leaf in kernel._leaves)
    if isinstance(kernel, kern_mod.Prod):
        return any(_contains_coregionalize(f) for f in kernel._factors)
    return False


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
    """FTC regression; ``str``/``repr`` use a paramz-inspired table.

    Default is Gaussian + exact FTC. Non-Gaussian CIP-003A families need
    ``inference='laplace'`` (e.g. ``likelihood=gpc.likelihoods.StudentT(...)``).
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: Any = "rbf",
        verbosity: int = 0,
        arccos_order: int = 0,
        likelihood: Any = "gaussian",
        inference: str = "exact",
    ) -> None:
        spec = kern_mod.normalize_kernel_arg(kernel)
        super().__init__(
            X,
            Y,
            kernel=spec,
            verbosity=verbosity,
            arccos_order=arccos_order,
            likelihood=likelihood,
            inference=inference,
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


def _emit_hermite_warnings(model: Any) -> None:
    """Surface the CIP-003D layout warnings (W1-W4) as ``UserWarning``.

    The text is generated once in C++; ``warnings.warn`` is used unconditionally
    because C++ stderr is easy to miss in a notebook.
    """
    for text in model.hermite_warnings():
        warnings.warn(text, UserWarning, stacklevel=3)


class SparseGPRegression(_SparseGPRegression):
    """Sparse Gaussian regression; ``str``/``repr`` use a paramz-inspired table.

    ``inducing`` is ``'points'`` (default), ``'weightspace'`` or ``'hermite'``.
    ``'hermite'`` (CIP-003D, ``kernel='mlp'`` only) uses fixed Hermite ridge
    windows with no free parameters. Leaving ``inducing_fixed`` unset resolves to
    fixed for Hermite (and ``str(model)`` says so); an explicit
    ``inducing_fixed=False`` raises, as do ``window_width`` and
    ``learn_window_width(s)``, and ``Z`` rows above the capacity for the input
    dimension. Conditioning, near-capacity and low-captured-variance cases emit a
    ``UserWarning``; ``describe()`` always shows the delivered layout.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        kernel: Any = "rbf",
        approx: str = "dtcvar",
        inducing_fixed: Optional[bool] = None,
        verbosity: int = 0,
        inducing: str = "points",
        window_width: Any = -1.0,
        arccos_order: int = 0,
        learn_window_width: bool = False,
        learn_window_widths: bool = False,
        sparse_linalg: str = "lq",
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
            sparse_linalg=sparse_linalg,
        )
        _emit_hermite_warnings(self)

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


class GPCoregionalizedRegression(GPRegression):
    """Multi-output ICM/LCM regression (GPy ``GPCoregionalizedRegression`` spirit).

    Accepts per-output ``X_list`` / ``Y_list``, stacks via ``build_XY``, and builds
    an ICM (or uses a supplied ICM/LCM kernel) plus optional ``IndexWhite``.
    Spherical ``likelihood.variance`` is frozen tiny when ``indexwhite`` is on
    (CIP-000D / ``CCoregionalizedRegression``).
    """

    def __init__(
        self,
        X_list: Sequence[np.ndarray],
        Y_list: Sequence[np.ndarray],
        kernel: Any,
        rank: int = 1,
        *,
        indexwhite: bool = True,
        verbosity: int = 0,
    ) -> None:
        if not isinstance(kernel, kern_mod.Kern):
            raise TypeError(
                "GPCoregionalizedRegression requires a gpc.kern.Kern "
                "(spatial kernel or ICM/LCM composition)"
            )
        X, Y = mo.build_XY(X_list, Y_list)
        q = int(np.asarray(X_list[0], dtype=float).shape[1])
        D = len(X_list)
        if _contains_coregionalize(kernel):
            k: kern_mod.Kern = kernel
        else:
            k = mo.ICM(q, D, kernel, W_rank=int(rank))
        if indexwhite and not _contains_indexwhite(k):
            k = k + kern_mod.IndexWhite(
                1, output_dim=D, variance=0.1, active_dims=[q]
            )
        self._num_outputs = D
        self._input_dim = q
        super().__init__(X, Y, kernel=k, verbosity=verbosity)

    @property
    def num_outputs(self) -> int:
        """Number of outputs (length of ``X_list``)."""
        return self._num_outputs

    def predict_output(
        self,
        Xstar: np.ndarray,
        output: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict for one output index (stamps the last column of ``Xstar``)."""
        Xs = np.asarray(Xstar, dtype=float)
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        if Xs.shape[1] == self._input_dim:
            Xs = np.hstack(
                [Xs, np.zeros((Xs.shape[0], 1), dtype=float)]
            )
        Xs = mo.stamp_output_index(Xs, int(output))
        return self.predict(Xs)


def _contains_indexwhite(kernel: kern_mod.Kern) -> bool:
    if isinstance(kernel, kern_mod.IndexWhite):
        return True
    if isinstance(kernel, kern_mod.Add):
        return any(_contains_indexwhite(leaf) for leaf in kernel._leaves)
    if isinstance(kernel, kern_mod.Prod):
        return any(_contains_indexwhite(f) for f in kernel._factors)
    return False


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





class DeepGP(_DeepGP):
    """Deep GP by doubly stochastic VI on stacked ``CSvgp`` layers (CIP-0035).

    Parameters
    ----------
    X, Y :
        Training inputs / outputs (NumPy arrays).
    layers :
        Number of GP layers ``L`` (default 2).
    hidden :
        Intermediate width (default ``X.shape[1]``).
    num_inducing :
        Int or length-``L`` sequence of inducing counts.
    kernel :
        ``'rbf'``, ``'mlp'``, ``'lin'``, ``'rbfard'`` or ``'mlpard'``: a string (same type on every layer)
        or a length-``layers`` sequence (one per layer, CIP-0041).
    inducing :
        ``'points'``, ``'weightspace'`` or ``'hermite'`` (the last two need
        ``kernel='mlp'`` on that layer): a string for every layer or a
        length-``layers`` sequence, e.g. ``['weightspace', 'points', 'points']``.
        Invalid combinations raise with the layer index. ``'hermite'`` uses fixed Hermite ridge windows
        (CIP-003D): no width, no offset, no learnable inducing parameters. It
        raises if ``num_inducing`` exceeds the capacity for the layer's input
        dimension, and warns (``UserWarning``) about conditioning, near-capacity
        fill, low captured variance and use in hidden layers; see
        ``describe()`` for the delivered layout.
    chi, beta :
        Hidden variance scale and observation precision init.
    drop_residual :
        If True, omit Nyström residual (finite-network bound).
    likelihood :
        ``'gaussian'``, ``'probit'``, ``'ordered'``, ``'robustmax'`` or
        ``'softmax'``, or a :class:`gpc.likelihoods.RobustMax` /
        :class:`gpc.likelihoods.Softmax` instance. The multi-class likelihoods take
        integer labels ``0..K-1`` (or one-hot ``N x K``) as ``Y``; the output
        dimension becomes ``K``. Use ``predict_proba`` for class probabilities.
    num_categories :
        Categories for ordered likelihood.
    num_classes :
        Number of classes for the multi-class likelihoods (default: inferred from
        the labels, or taken from a likelihood instance).
    learn_window_width, learn_window_widths :
        Weight-space only (``inducing='hermite'`` raises: the windows have no
        width); with per-layer ``inducing`` they apply to the weight-space
        layers. Optimise a tied window width ``s`` per layer, or untied
        per-unit widths ``s_m`` (mutually exclusive; see
        :class:`SparseGPRegression`).

    Notes
    -----
    ``predict`` returns the variance of a new observation (for a Gaussian
    likelihood it includes the noise ``1/beta``); ``predict_noiseless`` returns
    the latent-function variance. Both take ``samples`` and ``seed``.

    ``hidden`` may exceed the input dimension: the PCA mean path then uses all
    input directions and zero-pads the remaining hidden units, which the GP
    layer alone drives.

    Repeated ``optimize`` calls continue the same Adam state (moments and step
    count); call ``reset_optimizer`` to restart it.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        layers: int = 2,
        hidden: int = -1,
        num_inducing: Any = 10,
        kernel: Any = "rbf",
        inducing: Any = "points",
        chi: float = 1.0,
        beta: float = 1.0,
        drop_residual: bool = False,
        likelihood: Any = "gaussian",
        num_categories: int = 3,
        verbosity: int = 0,
        learn_window_width: bool = False,
        learn_window_widths: bool = False,
        num_classes: int = 0,
    ) -> None:
        mc: dict[str, Any] = {}
        if not isinstance(likelihood, str):
            from . import likelihoods as _lik

            if isinstance(likelihood, _lik.RobustMax):
                mc = {"likelihood_eps": likelihood.eps}
            elif isinstance(likelihood, _lik.Softmax):
                mc = {
                    "likelihood_samples": likelihood.num_samples,
                    "likelihood_seed": likelihood.seed,
                }
            else:
                raise TypeError(
                    "DeepGP likelihood must be a string, RobustMax or Softmax; got "
                    f"{type(likelihood).__name__}"
                )
            num_classes = int(likelihood.num_classes)
            likelihood = likelihood.type
        super().__init__(
            X,
            Y,
            layers=layers,
            hidden=hidden,
            num_inducing=num_inducing,
            kernel=kernel,
            inducing=inducing,
            chi=chi,
            beta=beta,
            drop_residual=drop_residual,
            likelihood=likelihood,
            num_categories=num_categories,
            verbosity=verbosity,
            learn_window_width=learn_window_width,
            learn_window_widths=learn_window_widths,
            num_classes=num_classes,
            **mc,
        )
        _emit_hermite_warnings(self)

    def __str__(self) -> str:  # noqa: D105
        return format_model_description(
            self.describe(),
            name="Deep GP",
            objective=self.objective(),
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()

    @classmethod
    def load(cls, path: str) -> Any:
        """Load a ``deepgp`` ``.model`` stream.

        Returns the bound C++ ``DeepGP`` instance (same methods as
        :class:`DeepGP` for ``elbo`` / ``predict`` / ``save``; construct a
        fresh :class:`DeepGP` when you need the Python ``optimize`` wrapper).
        """
        return _DeepGP.load(path)

    def optimize(
        self,
        iters: int = 100,
        batch_size: int = -1,
        samples: int = 1,
        adam_lr: float = 1e-2,
        natgrad_gamma: float = 0.0,
        seed: int = 0,
        messages: bool = False,
        callback: Callable[[OptimiserEvent], Any] | None = None,
    ) -> None:
        """Adam training (maximises ELBO); ``messages=True`` uses the paramz table."""
        _DeepGP.optimize(
            self,
            iters=iters,
            batch_size=batch_size,
            samples=samples,
            adam_lr=adam_lr,
            natgrad_gamma=natgrad_gamma,
            seed=seed,
            messages=False,
            callback=_build_callback(
                self,
                messages=messages,
                callback=callback,
                display_name="Deep GP",
                max_iters=iters,
            ),
        )

__all__ = [
    "GPRegression",
    "SparseGPRegression",
    "GPClassification",
    "GPCoregionalizedRegression",
    "GPLVM",
    "BayesianGPLVM",
    "MRD",
    "DeepGP",
]
