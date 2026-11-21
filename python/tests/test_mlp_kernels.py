"""CIP-001D Phase F: MLP / arc-cosine FTC façade smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy():
    rng = np.random.default_rng(2)
    X = np.linspace(-1.0, 1.0, 16)[:, None]
    Y = np.sin(2.0 * X) + 0.05 * rng.normal(size=X.shape)
    return X, Y


def test_mlp_construct_paths_optimize_predict():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y, kernel="mlp")
    assert m.kernel_type == "mlp"
    assert m.approx == "ftc"
    paths = m.param_paths()
    assert any("mlp" in p for p in paths)
    # GPy-facing alias (CIP-0014)
    m.set_by_path("mlp.weight_variance", 5.0)
    assert m.get_by_path("mlp.weight_variance") == pytest.approx(5.0)
    m.set_by_path("mlp.bias_variance", 2.0)
    m.set_by_path("mlp.variance", 1.25)
    before = m.objective()
    assert np.isfinite(before)
    m.optimize(iters=20)
    assert m.objective() <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert var.shape == Y.shape
    assert np.all(var >= 0.0)


def test_mlp_string_kernel_scales_weight_variance_to_input():
    # Calendar-scale X: default weight_variance must be O(1/Var(X)), else arcsin saturates.
    X = np.linspace(1896.0, 2024.0, 20)[:, None]
    Y = np.linspace(3.0, 4.5, 20)[:, None] + 0.05
    m = gpc.GPRegression(X, Y, kernel="mlp")
    expected = 1.0 / float(np.var(X))
    assert m.get_by_path("mlp.weight_variance") == pytest.approx(expected, rel=1e-6)
    assert m.get_by_path("mlp.bias_variance") == pytest.approx(1.0)
    m.optimize(iters=80)
    Xtest = np.linspace(1890.0, 2030.0, 100)[:, None]
    mu, _ = m.predict(Xtest)
    assert float(mu.max() - mu.min()) > 0.5


def test_arccos_order_zero_and_one():
    X, Y = _toy()
    m0 = gpc.GPRegression(X, Y, kernel="arccos", arccos_order=0)
    assert m0.kernel_type == "arccos"
    # param_paths uses display name ArcCosine.*; type alias arccos.* also resolves.
    assert any("ArcCosine." in p for p in m0.param_paths())
    m0.set_by_path("arccos.variance", 1.5)
    assert m0.get_by_path("ArcCosine.variance") == pytest.approx(1.5)
    assert np.isfinite(m0.objective())

    m1 = gpc.GPRegression(X, Y, kernel="arccosine", arccos_order=1)
    assert m1.kernel_type == "arccos"
    before = m1.objective()
    m1.optimize(iters=15)
    assert m1.objective() <= before + 1e-6


def test_bad_arccos_order_raises():
    X, Y = _toy()
    with pytest.raises(gpc.GpcError):
        gpc.GPRegression(X, Y, kernel="arccos", arccos_order=2)


def test_sparse_mlp_points():
    X, Y = _toy()
    Z = np.linspace(-0.8, 0.8, 5)[:, None]
    m = gpc.SparseGPRegression(X, Y, Z, kernel="mlp")
    assert m.kernel_type == "mlp"
    assert m.inducing_type == "points"
    before = m.objective()
    m.optimize(iters=15)
    assert m.objective() <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape and np.all(var >= 0.0)
