"""CIP-0040 Phase 1d: gpc.kern.Conv façade and DeepGP integration."""

from __future__ import annotations

import numpy as np
import pytest

import gpc
from gpc.kern import MLP, RBF, Conv

H, W, C = 6, 6, 1
PH, PW = 3, 3


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, H * W * C))
    y = np.sin(X[:, :9].sum(1, keepdims=True)) + 0.05 * rng.normal(size=(n, 1))
    return X, y


def _double_loop(X, X2, ell, var, stride, w):
    def patches(x):
        img = x.reshape(H, W, C)
        out = []
        for i in range(0, H - PH + 1, stride):
            for j in range(0, W - PW + 1, stride):
                out.append(img[i : i + PH, j : j + PW, :].reshape(-1))
        return np.array(out)

    K = np.zeros((len(X), len(X2)))
    for a, xa in enumerate(X):
        pa = patches(xa)
        for b, xb in enumerate(X2):
            pb = patches(xb)
            d = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
            K[a, b] = (w[:, None] * w[None, :] * var * np.exp(-0.5 * d / ell**2)).sum()
    return K


@pytest.mark.parametrize("stride", [1, 2])
def test_conv_K_matches_double_loop(stride):
    X, _ = _data(5)
    k = Conv(RBF(PH * PW * C, variance=1.7, lengthscale=1.3), (H, W, C), (PH, PW),
             stride=stride, weights="fixed")
    P = k.num_patches
    ref = _double_loop(X, X[:3], 1.3, 1.7, stride, np.full(P, 1.0 / P))
    np.testing.assert_allclose(k.K(X, X[:3]), ref, rtol=1e-10)
    Ks = k.K(X)
    np.testing.assert_allclose(Ks, Ks.T, atol=1e-12)


def test_conv_validation():
    base = RBF(PH * PW * C)
    with pytest.raises(ValueError, match="h\\*w\\*C"):
        Conv(RBF(5), (H, W, C), (PH, PW))
    with pytest.raises(ValueError, match="fit inside"):
        Conv(base, (H, W, C), (7, 3))
    with pytest.raises(ValueError, match="weights"):
        Conv(base, (H, W, C), (PH, PW), weights="bogus")
    with pytest.raises(ValueError):
        Conv(base, (H, W), (PH, PW))
    k = Conv(base, (H, W, C), (PH, PW))
    with pytest.raises(Exception, match="columns"):
        k.K(np.zeros((2, 7)))


def test_cg_models_refuse_conv():
    X, y = _data(10)
    k = Conv(RBF(PH * PW * C), (H, W, C), (PH, PW))
    with pytest.raises(NotImplementedError, match="DeepGP"):
        gpc.models.GPRegression(X, y, kernel=k)


def test_deepgp_conv_trains_and_exposes_filters():
    X, y = _data(40)
    k = Conv(RBF(PH * PW * C, lengthscale=2.0), (H, W, C), (PH, PW), stride=1, weights="tied")
    m = gpc.models.DeepGP(X, y, layers=1, kernel=k, num_inducing=8)
    f0 = m.inducing_filters().copy()
    assert f0.shape == (8, PH, PW, C)
    e0 = m.elbo() if hasattr(m, "elbo") else None
    m.optimize(iters=40, batch_size=20, adam_lr=0.02, seed=1)
    assert not np.allclose(f0, m.inducing_filters())
    if e0 is not None:
        assert m.elbo() > e0
    mu, var = m.predict(X[:5])
    assert np.all(np.isfinite(mu)) and np.all(var > 0)
    assert m.patch_weights().shape == (k.num_patches,)


def test_deepgp_conv_two_layers_and_errors():
    X, y = _data(30)
    k = Conv(RBF(PH * PW * C), (H, W, C), (PH, PW), stride=2)
    m = gpc.models.DeepGP(X, y, layers=2, hidden=4, kernel=[k, "rbf"], num_inducing=6)
    m.optimize(iters=5, batch_size=15, seed=0)
    with pytest.raises(Exception, match="first layer only"):
        gpc.models.DeepGP(X, y, layers=2, hidden=H * W * C, kernel=["rbf", k])
    with pytest.raises(Exception, match="requires a convolutional kernel"):
        gpc.models.DeepGP(X, y, layers=1, kernel="rbf", inducing="patches")
    with pytest.raises(Exception, match="expects|columns|input columns"):
        gpc.models.DeepGP(X[:, :20], y, layers=1,
                          kernel=Conv(RBF(PH * PW * C), (H, W, C), (PH, PW)))


def test_diag_estimate_guards():
    X, y = _data(30)
    k = Conv(RBF(PH * PW * C), (H, W, C), (PH, PW))
    with pytest.raises(Exception, match="unbiased only"):
        gpc.models.DeepGP(X, y, layers=2, hidden=4, kernel=[k, "rbf"], diag="estimate")
    with pytest.raises(Exception, match="unbiased only"):
        gpc.models.DeepGP(X, (y > 0).astype(float), layers=1, kernel=k,
                          likelihood="probit", diag="estimate")
    m = gpc.models.DeepGP(X, y, layers=1, kernel=k, num_inducing=6, diag="estimate", diag_samples=8)
    m.optimize(iters=5, batch_size=15, seed=0)
    gpc.models.DeepGP(X, y, layers=2, hidden=4, kernel=[k, "rbf"], diag="estimate",
                      diag_force=True)
    with pytest.raises(Exception, match="convolutional"):
        gpc.models.DeepGP(X, y, layers=1, kernel="rbf", diag="estimate")
