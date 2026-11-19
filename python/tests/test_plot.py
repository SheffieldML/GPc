"""Optional plot helper (skipped if matplotlib missing)."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import gpc
from gpc.plot import plot_1d


def test_plot_1d_smoke():
    X = np.linspace(0.0, 1.0, 12)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05
    m = gpc.GPRegression(X, Y)
    m.optimize(iters=10)
    ax, mu, var = plot_1d(m, X, Y)
    assert mu.shape[0] == 200
    assert var.shape[0] == 200
    assert ax is not None
