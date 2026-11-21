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


def test_plot_1d_noiseless_narrower_than_predict():
    X = np.linspace(0.0, 1.0, 12)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05
    m = gpc.GPRegression(X, Y)
    m.set_by_path("likelihood.variance", 0.04)
    Xtest = np.linspace(0.0, 1.0, 50)[:, None]
    _, mu_y, var_y = plot_1d(m, X, Y, Xtest=Xtest, noiseless=False)
    _, mu_f, var_f = plot_1d(m, X, Y, Xtest=Xtest, noiseless=True)
    assert np.allclose(mu_y, mu_f)
    assert np.allclose(var_y - var_f, 0.04, atol=1e-12)
