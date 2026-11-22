"""Kern.K / thin plot — backlog 2026-11-22_cip001D-kern-K-and-plot."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _rbf_numpy(X, X2, variance=1.0, lengthscale=1.0):
    X = np.asarray(X, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    d2 = np.sum((X[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return variance * np.exp(-0.5 * d2 / (lengthscale**2))


def test_rbf_K_matches_numpy():
    X = np.linspace(0.0, 1.0, 8)[:, None]
    k = gpc.kern.RBF(1, variance=1.2, lengthscale=0.4)
    K = k.K(X)
    want = _rbf_numpy(X, X, variance=1.2, lengthscale=0.4)
    assert K.shape == (8, 8)
    np.testing.assert_allclose(K, want, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(K, K.T, rtol=0, atol=1e-12)


def test_K_cross_and_prod():
    X = np.linspace(0.0, 1.0, 5)[:, None]
    X2 = np.linspace(-0.2, 0.8, 3)[:, None]
    k = gpc.kern.RBF(1, variance=0.8, lengthscale=0.5) * gpc.kern.Linear(
        1, variance=0.4
    )
    K = k.K(X, X2)
    assert K.shape == (5, 3)
    # Product: elementwise K_rbf * K_lin
    Kr = gpc.kern.RBF(1, variance=0.8, lengthscale=0.5).K(X, X2)
    Kl = gpc.kern.Linear(1, variance=0.4).K(X, X2)
    np.testing.assert_allclose(K, Kr * Kl, rtol=1e-6, atol=1e-6)


def test_K_add_and_active_dims():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 2))
    k = gpc.kern.RBF(1, variance=1.0, lengthscale=0.7, active_dims=[1]) + gpc.kern.Bias(
        1, variance=0.3, active_dims=[1]
    )
    K = k.K(X)
    assert K.shape == (6, 6)
    # Bias contributes constant 0.3; RBF uses column 1 only.
    Kb = np.full((6, 6), 0.3)
    Kr = _rbf_numpy(X[:, 1:2], X[:, 1:2], variance=1.0, lengthscale=0.7)
    np.testing.assert_allclose(K, Kr + Kb, rtol=1e-6, atol=1e-6)


def test_plot_kern_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    k = gpc.kern.RBF(1, lengthscale=0.5)
    ax = k.plot(plot_limits=(-1.0, 1.0), resolution=50)
    assert ax is not None
