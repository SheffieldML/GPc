"""CIP-001D Phase B: FTC GPRegression smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy():
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 12)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    return X, Y


def test_construct_and_paths():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    assert m.n == 12
    assert m.input_dim == 1
    paths = m.param_paths()
    assert any("rbf" in p for p in paths)
    m.set_by_path("rbf.lengthscale", 0.5)
    assert m.get_by_path("rbf.lengthscale") == pytest.approx(0.5)


def test_optimize_and_predict():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    before = m.objective()
    m.optimize(iters=20)
    after = m.objective()
    assert np.isfinite(after)
    # SCG should not make the objective worse on this toy (allow tiny noise).
    assert after <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert var.shape == Y.shape
    assert np.all(var >= 0.0)


def test_shape_mismatch():
    X = np.zeros((5, 1))
    Y = np.zeros((4, 1))
    with pytest.raises(gpc.GpcError):
        gpc.GPRegression(X, Y)
