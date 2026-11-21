"""CIP-001D follow-on: MAP GPLVM and collapsed BayesianGPLVM smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy_y(n: int = 20, d: int = 4, q: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, q))
    W = rng.normal(size=(q, d))
    Y = X @ W + 0.05 * rng.normal(size=(n, d))
    return Y


def test_gplvm_construct_optimize_predict():
    Y = _toy_y()
    m = gpc.models.GPLVM(Y, input_dim=2)
    assert m.n == 20
    assert m.input_dim == 2
    assert m.output_dim == 4
    assert "rbf" in m.kernel_type
    X = m.latent_mean()
    assert X.shape == (20, 2)
    before = m.objective()
    m.optimize(iters=15)
    assert m.objective() <= before + 1e-5
    mu, var = m.predict(X[:3])
    assert mu.shape == (3, 4)
    assert var.shape == (3, 4)
    assert np.all(var > 0.0)
    mu2, var2 = m.predict_noiseless(X[:3])
    np.testing.assert_allclose(mu, mu2)
    np.testing.assert_allclose(var, var2)


def test_gplvm_parts_kernel():
    Y = _toy_y(n=16)
    k = gpc.kern.RBF(2) + gpc.kern.White(2)
    m = gpc.models.GPLVM(Y, input_dim=2, kernel=k)
    assert "rbf" in m.kernel_type
    assert "white" in m.kernel_type
    m.optimize(iters=5)


def test_bayesian_gplvm_construct_optimize_predict():
    Y = _toy_y(n=24, d=5)
    m = gpc.models.BayesianGPLVM(Y, input_dim=2, num_inducing=8)
    assert m.n == 24
    assert m.input_dim == 2
    assert m.num_inducing == 8
    assert m.kernel_type == "rbfard"
    assert m.approx == "vardtc"
    mu_x = m.latent_mean()
    s_x = m.latent_variance()
    z = m.inducing_inputs()
    assert mu_x.shape == (24, 2)
    assert s_x.shape == (24, 2)
    assert z.shape == (8, 2)
    assert np.all(s_x > 0.0)
    before = m.objective()
    m.optimize(iters=20)
    assert m.objective() <= before + 1e-4
    assert m.elbo() == pytest.approx(-m.objective())
    mu, var = m.predict(mu_x[:4])
    mu_f, var_f = m.predict_noiseless(mu_x[:4])
    assert mu.shape == (4, 5)
    assert var.shape == (4, 5)
    # CIP-0031: predictive variance includes 1/β above latent.
    assert np.all(var >= var_f - 1e-12)


def test_bayesian_gplvm_isotropic_rbf():
    Y = _toy_y(n=18)
    m = gpc.models.BayesianGPLVM(Y, input_dim=2, num_inducing=6, ard=False)
    assert m.kernel_type == "rbf"
    m.optimize(iters=8)


def test_bayesian_gplvm_bad_inducing():
    Y = _toy_y(n=12)
    with pytest.raises(gpc.GpcError):
        gpc.models.BayesianGPLVM(Y, input_dim=2, num_inducing=0)
    with pytest.raises(gpc.GpcError):
        gpc.models.BayesianGPLVM(Y, input_dim=2, num_inducing=20)


def test_gplvm_dim_guard():
    Y = _toy_y(n=10, d=2, q=2)
    with pytest.raises(gpc.GpcError):
        gpc.models.GPLVM(Y, input_dim=3)
