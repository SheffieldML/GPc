"""CIP-0031: predict vs predict_noiseless (predictive = latent + likelihood.variance)."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy(n: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.linspace(-1.0, 1.0, n)[:, None]
    Y = np.sin(2.0 * np.pi * X) + 0.05 * rng.standard_normal((n, 1))
    return X, Y


def test_ftc_predict_minus_noiseless_equals_likelihood_variance():
    X, Y = _toy()
    k = gpc.kern.RBF(1)
    m = gpc.models.GPRegression(X, Y, kernel=k)
    sigma2 = 0.04
    m.set_by_path("likelihood.variance", sigma2)
    mu_p, var_p = m.predict(X)
    mu_l, var_l = m.predict_noiseless(X)
    assert mu_p.shape == Y.shape and var_p.shape == Y.shape
    assert np.allclose(mu_p, mu_l)
    assert np.allclose(var_p - var_l, sigma2, atol=1e-12)


def test_sparse_predict_minus_noiseless_equals_likelihood_variance():
    X, Y = _toy(n=16)
    Z = X[::4].copy()
    k = gpc.kern.RBF(1)
    m = gpc.models.SparseGPRegression(X, Y, Z, kernel=k, approx="dtcvar")
    sigma2 = 0.04
    m.set_by_path("likelihood.variance", sigma2)
    mu_p, var_p = m.predict(X)
    mu_l, var_l = m.predict_noiseless(X)
    assert np.allclose(mu_p, mu_l)
    assert np.allclose(var_p - var_l, sigma2, atol=1e-12)
