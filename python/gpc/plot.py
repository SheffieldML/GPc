"""Optional thin plotting helpers (`gpc[plot]` / matplotlib).

Not imported by `import gpc`. Use::

    from gpc.plot import plot_1d, plot_kern
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np


def plot_kern(
    kern: Any,
    ax: Any = None,
    x: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    plot_limits: Optional[Tuple[float, float]] = None,
    resolution: int = 200,
    label: Optional[str] = None,
) -> Any:
    """Thin GPy-style kernel slice plot: ``K(X_grid, x0)`` vs first free dim.

    Parameters
    ----------
    kern
        Object with ``K(X, X2)`` (``gpc.kern.Kern``).
    ax
        Matplotlib axes; created if omitted.
    x
        Fixed input location for the other columns / slice centre. Scalar
        fills all dims; length-``q`` vector sets ``x0`` fully. Default zeros.
    plot_limits
        ``(xmin, xmax)`` along the plotted dimension (index 0, or the sole
        ``active_dims`` entry when set). Default ``(-3, 3)``.
    resolution
        Number of grid points.
    label
        Optional legend label (defaults to ``repr(kern)``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gpc.plot requires matplotlib. Install with: pip install 'gpc[plot]'"
        ) from exc

    q = int(getattr(kern, "input_dim", 0) or 0)
    active = getattr(kern, "active_dims", None)
    if isinstance(kern, type(None)):
        raise TypeError("kern is required")
    # Add / Prod: infer dim from first leaf / factor.
    if q < 1:
        leaves = getattr(kern, "_leaves", None) or getattr(kern, "_factors", None)
        if leaves:
            q = int(getattr(leaves[0], "input_dim", 1))
            if active is None:
                active = getattr(leaves[0], "active_dims", None)
        else:
            q = 1
    plot_dim = int(active[0]) if active else 0
    if plot_limits is None:
        plot_limits = (-3.0, 3.0)
    xmin, xmax = float(plot_limits[0]), float(plot_limits[1])
    grid = np.linspace(xmin, xmax, int(resolution))

    if x is None:
        x0 = np.zeros((1, q), dtype=float)
    else:
        xv = np.asarray(x, dtype=float).reshape(-1)
        if xv.size == 1:
            x0 = np.full((1, q), float(xv[0]), dtype=float)
        elif xv.size == q:
            x0 = xv.reshape(1, q)
        else:
            raise ValueError(f"x must be scalar or length-{q}, got size {xv.size}")

    X = np.tile(x0, (grid.size, 1))
    X[:, plot_dim] = grid
    kvals = np.asarray(kern.K(X, x0)).reshape(-1)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(grid, kvals, label=label if label is not None else repr(kern))
    ax.set_xlabel(f"x[{plot_dim}]")
    ax.set_ylabel("k(x, x0)")
    ax.legend(loc="best")
    return ax


def plot_1d(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    Xtest: Optional[np.ndarray] = None,
    ax: Any = None,
    label: str = "mean",
    noiseless: bool = False,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    r"""Plot 1-D data and mean ± 2σ from ``predict`` or ``predict_noiseless``.

    Parameters
    ----------
    model
        Object with ``predict`` / ``predict_noiseless`` → ``(mu, var)``.
    X, Y
        Training inputs/targets (N×1).
    Xtest
        Grid for the predictive curve; defaults to a dense span of ``X``.
    ax
        Matplotlib axes; created if omitted.
    noiseless
        If False (default), call ``predict`` (\(y_*\), includes observation
        noise). If True, call ``predict_noiseless`` (\(f_*\); CIP-0031).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gpc.plot requires matplotlib. Install with: pip install 'gpc[plot]'"
        ) from exc

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Xtest is None:
        xmin, xmax = float(X.min()), float(X.max())
        pad = 0.05 * (xmax - xmin + 1e-12)
        Xtest = np.linspace(xmin - pad, xmax + pad, 200)[:, None]
    else:
        Xtest = np.asarray(Xtest, dtype=float)

    if noiseless:
        mu, var = model.predict_noiseless(Xtest)
    else:
        mu, var = model.predict(Xtest)
    mu = np.asarray(mu).reshape(-1)
    std = np.sqrt(np.maximum(np.asarray(var).reshape(-1), 0.0))

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(X.reshape(-1), Y.reshape(-1), "k.", label="data")
    ax.plot(Xtest.reshape(-1), mu, label=label)
    band = "±2σ (f_*)" if noiseless else "±2σ (y_*)"
    ax.fill_between(
        Xtest.reshape(-1),
        mu - 2.0 * std,
        mu + 2.0 * std,
        alpha=0.25,
        label=band,
    )
    ax.legend(loc="best")
    return ax, mu, var
