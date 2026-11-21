"""CIP-0033 / CIP-0036: GPClassification façade (GPy capability reflection)."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy():
    X = np.linspace(-1.5, 1.5, 8)[:, None]
    Y = (X[:, 0] < 0.0).astype(float)[:, None]
    return X, Y


def test_construct_ep_and_paths():
    X, Y = _toy()
    m = gpc.GPClassification(X, Y, inference="ep")
    assert m.n == 8
    assert m.approx == "ftc"
    assert m.inference == "ep"
    paths = m.param_paths()
    assert any("rbf" in p for p in paths)
    before = m.objective()
    assert np.isfinite(before)
    m.optimize(iters=15)
    after = m.objective()
    assert after <= before + 1e-6


def test_predict_proba_shape():
    X, Y = _toy()
    m = gpc.GPClassification(X, Y)
    m.set_by_path("rbf.lengthscale", 0.8)
    m.set_by_path("rbf.variance", 1.0)
    mu, var = m.predict_noiseless(X)
    assert mu.shape == (8, 1)
    assert var.shape == (8, 1)
    assert np.all(var > 0)
    p = m.predict_proba(X)
    assert p.shape == (8, 1)
    assert np.all((p > 0.0) & (p < 1.0))
    # GPy aliases (CIP-0036).
    mu2, var2 = m.predict(X)
    np.testing.assert_allclose(mu2, mu)
    np.testing.assert_allclose(var2, var)
    p2 = m.predictive_probabilities(X)
    np.testing.assert_allclose(p2, p)


def test_laplace_predict():
    X, Y = _toy()
    m = gpc.GPClassification(X, Y, inference="laplace")
    assert m.inference == "laplace"
    assert np.isfinite(m.objective())
    m.set_by_path("rbf.lengthscale", 0.8)
    m.set_by_path("rbf.variance", 1.0)
    mu, var = m.predict_noiseless(X)
    assert mu.shape == (8, 1)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)
    p = m.predict_proba(X)
    assert np.all((p > 0.0) & (p < 1.0))
    # Loose EP↔Laplace agreement on separable toy.
    m_ep = gpc.GPClassification(X, Y, inference="ep")
    m_ep.set_by_path("rbf.lengthscale", 0.8)
    m_ep.set_by_path("rbf.variance", 1.0)
    p_ep = m_ep.predict_proba(X)
    assert np.mean(np.abs(p - p_ep)) < 0.35


def test_sparse_inference_rejected():
    X, Y = _toy()
    with pytest.raises(gpc.GpcError):
        gpc.GPClassification(X, Y, inference="epdtc")
