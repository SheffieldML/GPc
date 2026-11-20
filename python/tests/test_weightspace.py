"""CIP-0028: SparseGPRegression weight-space inducing smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy():
    rng = np.random.default_rng(3)
    X = np.linspace(-1.5, 1.5, 24)[:, None]
    Y = np.sin(2.0 * X) + 0.1 * np.cos(3.0 * X) + 0.05 * rng.normal(size=X.shape)
    Z = np.linspace(-1.2, 1.2, 8)[:, None]
    return X, Y, Z


def test_mlp_weightspace_construct_optimize_predict():
    X, Y, Z = _toy()
    m = gpc.SparseGPRegression(
        X, Y, Z, kernel="mlp", inducing="weightspace", window_width=0.55
    )
    assert m.kernel_type == "mlp"
    assert m.inducing_type == "weightspace"
    assert m.approx == "dtcvar"
    assert m.num_inducing == 8
    assert m.window_width == pytest.approx(0.55)
    paths = m.param_paths()
    assert any("mlp" in p for p in paths)
    # No Bias in weight-space compounds (CIP-0028 Phase 3)
    assert not any("bias." in p for p in paths)
    before = m.objective()
    assert np.isfinite(before)
    m.optimize(iters=20)
    after = m.objective()
    assert np.isfinite(after)
    assert after <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert np.all(var >= 0.0)


def test_weightspace_rejects_rbf():
    X, Y, Z = _toy()
    with pytest.raises(gpc.GpcError, match="weightspace"):
        gpc.SparseGPRegression(X, Y, Z, kernel="rbf", inducing="weightspace")


def test_arccos_weightspace_smoke():
    X, Y, Z = _toy()
    m = gpc.SparseGPRegression(
        X, Y, Z, kernel="arccos", inducing="weightspace", arccos_order=0
    )
    assert m.inducing_type == "weightspace"
    assert np.isfinite(m.objective())
    mu, var = m.predict(X)
    assert mu.shape == Y.shape and np.all(var >= 0.0)


def test_unknown_inducing_raises():
    X, Y, Z = _toy()
    with pytest.raises(gpc.GpcError, match="inducing"):
        gpc.SparseGPRegression(X, Y, Z, inducing="greens")
