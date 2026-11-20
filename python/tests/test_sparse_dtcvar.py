"""CIP-001D Phase C: SparseGPRegression DTCVAR smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy_sparse():
    rng = np.random.default_rng(1)
    X = np.linspace(0.0, 1.0, 24)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    Z = np.linspace(0.05, 0.95, 6)[:, None]
    return X, Y, Z


def test_construct_dtcvar_and_paths():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(X, Y, Z)
    assert m.approx == "dtcvar"
    assert m.num_inducing == 6
    assert m.inducing_fixed is True
    assert m.n == 24
    paths = m.param_paths()
    assert any("rbf" in p for p in paths)
    m.set_by_path("rbf.lengthscale", 0.4)
    assert m.get_by_path("rbf.lengthscale") == pytest.approx(0.4)
    np.testing.assert_allclose(m.inducing_inputs(), Z)


def test_optimize_and_predict():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(X, Y, Z)
    before = m.objective()
    m.optimize(iters=20)
    after = m.objective()
    assert np.isfinite(after)
    assert after <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert var.shape == Y.shape
    assert np.all(var >= 0.0)


def test_fitc_and_dtc_smoke():
    X, Y, Z = _toy_sparse()
    m_fitc = gpc.SparseGPRegression(X, Y, Z, approx="fitc")
    assert m_fitc.approx == "fitc"
    before = m_fitc.objective()
    m_fitc.optimize(iters=15)
    assert m_fitc.objective() <= before + 1e-6
    mu, var = m_fitc.predict(X)
    assert mu.shape == Y.shape and np.all(var >= 0.0)

    m_dtc = gpc.SparseGPRegression(X, Y, Z, approx="dtc")
    assert m_dtc.approx == "dtc"
    assert np.isfinite(m_dtc.objective())


def test_unknown_approx_raises():
    X, Y, Z = _toy_sparse()
    with pytest.raises(gpc.GpcError, match="not supported"):
        gpc.SparseGPRegression(X, Y, Z, approx="pitc")


def test_z_shape_mismatch():
    X, Y, Z = _toy_sparse()
    with pytest.raises(gpc.GpcError):
        gpc.SparseGPRegression(X, Y, np.zeros((6, 2)))
    with pytest.raises(gpc.GpcError):
        gpc.SparseGPRegression(X, Y, np.linspace(0, 1, 30)[:, None])
