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
    assert m.kernel_type == "rbf"
    paths = m.param_paths()
    assert any("rbf" in p for p in paths)
    m.set_by_path("rbf.lengthscale", 0.5)
    assert m.get_by_path("rbf.lengthscale") == pytest.approx(0.5)


def test_expquad_kernel_paths():
    """CIP-0024: GPy ExpQuad lengthscale-first surface via kernel='eq'."""
    X, Y = _toy()
    m = gpc.GPRegression(X, Y, kernel="ExpQuad")
    assert m.kernel_type == "eq"
    paths = m.param_paths()
    assert any("ExpQuad." in p for p in paths)
    assert not any("rbf." in p for p in paths)
    # Model paths are kern.ExpQuad.*; bare ExpQuad.* is also accepted.
    m.set_by_path("ExpQuad.lengthscale", 0.5)
    assert m.get_by_path("ExpQuad.lengthscale") == pytest.approx(0.5)
    assert m.get_by_path("kern.ExpQuad.lengthscale") == pytest.approx(0.5)
    m.set_by_path("ExpQuad.variance", 1.25)
    assert m.get_by_path("ExpQuad.variance") == pytest.approx(1.25)
    before = m.objective()
    m.optimize(iters=15)
    assert m.objective() <= before + 1e-6


def test_unknown_kernel_raises():
    X, Y = _toy()
    with pytest.raises(gpc.GpcError):
        gpc.GPRegression(X, Y, kernel="matern32")


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
