"""Optional thin plotting helpers (`gpc[plot]` / matplotlib).

Not imported by `import gpc`. Use::

    from gpc.plot import plot_1d
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def plot_1d(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    Xtest: Optional[np.ndarray] = None,
    ax: Any = None,
    label: str = "mean",
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Plot 1-D data and predictive mean ± 2σ from ``model.predict``.

    Parameters
    ----------
    model
        Object with ``predict(Xtest) -> (mu, var)``.
    X, Y
        Training inputs/targets (N×1).
    Xtest
        Grid for the predictive curve; defaults to a dense span of ``X``.
    ax
        Matplotlib axes; created if omitted.
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

    mu, var = model.predict(Xtest)
    mu = np.asarray(mu).reshape(-1)
    std = np.sqrt(np.maximum(np.asarray(var).reshape(-1), 0.0))

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(X.reshape(-1), Y.reshape(-1), "k.", label="data")
    ax.plot(Xtest.reshape(-1), mu, label=label)
    ax.fill_between(
        Xtest.reshape(-1),
        mu - 2.0 * std,
        mu + 2.0 * std,
        alpha=0.25,
        label="±2σ",
    )
    ax.legend(loc="best")
    return ax, mu, var
