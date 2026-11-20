"""CIP-001D: gpc.kern factories, compound +, kernel= object API."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy():
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 12)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    return X, Y


def test_kern_add_flattens():
    k = gpc.kern.RBF(1, lengthscale=0.2) + gpc.kern.Bias(1) + gpc.kern.White(1, variance=0.05)
    parts = k.to_parts()
    assert [p["type"] for p in parts] == ["rbf", "bias", "white"]
    assert parts[0]["lengthscale"] == pytest.approx(0.2)
    assert parts[2]["variance"] == pytest.approx(0.05)


def test_gp_regression_with_kern_object():
    X, Y = _toy()
    k = gpc.kern.RBF(1, lengthscale=0.5, variance=1.0) + gpc.kern.Bias(1, variance=0.1) + gpc.kern.White(
        1, variance=0.05
    )
    m = gpc.models.GPRegression(X, Y, kernel=k)
    assert "rbf" in m.kernel_type
    assert "white" in m.kernel_type
    assert m.get_by_path("rbf.lengthscale") == pytest.approx(0.5)
    assert m.get_by_path("bias.variance") == pytest.approx(0.1)
    assert m.get_by_path("white.variance") == pytest.approx(0.05)
    before = m.objective()
    m.optimize(iters=15)
    assert m.objective() <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert np.all(var >= 0.0)


def test_package_level_accepts_kern_object():
    X, Y = _toy()
    k = gpc.kern.ExpQuad(1) + gpc.kern.White(1)
    m = gpc.GPRegression(X, Y, kernel=k)
    paths = m.param_paths()
    assert any("ExpQuad." in p for p in paths)
    assert not any("bias." in p for p in paths)


def test_sparse_with_kern_object():
    rng = np.random.default_rng(1)
    X = np.linspace(0.0, 1.0, 24)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    Z = np.linspace(0.05, 0.95, 6)[:, None]
    k = gpc.kern.RBF(1) + gpc.kern.White(1)
    m = gpc.models.SparseGPRegression(X, Y, Z, kernel=k)
    assert m.approx == "dtcvar"
    assert any("rbf" in p for p in m.param_paths())
    m.optimize(iters=10)
    mu, var = m.predict(X)
    assert mu.shape == Y.shape


def test_string_kernel_default_signal_only():
    """CIP-0027: string kernels are signal-only; noise is likelihood.variance."""
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    paths = m.param_paths()
    assert any("rbf" in p for p in paths)
    assert any(p == "likelihood.variance" or p.endswith("likelihood.variance") for p in paths)
    assert not any("bias" in p for p in paths)
    assert not any("white" in p for p in paths)
