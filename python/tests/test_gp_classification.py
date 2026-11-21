"""CIP-0033 Phase D: GPClassification façade smoke."""

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


def test_laplace_objective():
    X, Y = _toy()
    m = gpc.GPClassification(X, Y, inference="laplace")
    assert m.inference == "laplace"
    assert np.isfinite(m.objective())


def test_sparse_inference_rejected():
    X, Y = _toy()
    with pytest.raises(gpc.GpcError):
        gpc.GPClassification(X, Y, inference="epdtc")
