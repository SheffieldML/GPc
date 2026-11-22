"""CIP-0035: gpc.models.DeepGP smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gpc


def _sinc_toy(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.linspace(-2.0, 2.0, n)[:, None]
    Y = np.sin(1.5 * X) + 0.05 * rng.normal(size=X.shape)
    return X, Y


def test_construct_and_elbo():
    X, Y = _sinc_toy()
    m = gpc.models.DeepGP(X, Y, layers=2, hidden=1, num_inducing=6, chi=0.5)
    assert m.n == X.shape[0]
    assert m.num_layers == 2
    assert m.approx == "deepgp"
    assert m.inducing_type == "points"
    elbo = m.elbo()
    assert np.isfinite(elbo)


def test_optimize_and_predict():
    X, Y = _sinc_toy()
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=1, num_inducing=6, chi=0.5, beta=5.0
    )
    before = m.elbo()
    m.optimize(iters=15, batch_size=X.shape[0], adam_lr=0.05, seed=3)
    after = m.elbo()
    assert np.isfinite(after)
    # Short Adam run should not explode; allow a small drop under noise.
    assert after >= before - 5.0
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert var.shape == Y.shape
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0.0)
    mu_s, var_s = m.predict(X, samples=8, seed=5)
    assert mu_s.shape == Y.shape
    assert np.all(np.isfinite(mu_s))
    assert np.all(var_s > 0.0)


def test_jacobian_spread_prior_draws():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(12, 2))
    Y = rng.normal(size=(12, 2))
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=2, num_inducing=6, chi=0.5, beta=1.0
    )
    spread = m.jacobian_spread(n_probe=8, probe_seed=11)
    assert np.isfinite(spread)
    assert 0.05 < spread < 5.0


def test_probit_likelihood():
    rng = np.random.default_rng(1)
    X = np.linspace(-1.5, 1.5, 20)[:, None]
    Y = (np.sin(X) > 0.0).astype(float) * 2.0 - 1.0
    Y += 0.0 * rng.normal(size=Y.shape)
    m = gpc.models.DeepGP(
        X, Y, layers=1, num_inducing=5, likelihood="probit", beta=1.0
    )
    assert m.likelihood_type == "probit"
    assert np.isfinite(m.elbo())
    m.optimize(iters=8, seed=2, adam_lr=0.05)
    assert np.isfinite(m.elbo())


def test_save_load_roundtrip(tmp_path: Path):
    X, Y = _sinc_toy(n=16)
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=1, num_inducing=4, chi=0.4, drop_residual=True
    )
    m.optimize(iters=5, seed=4, adam_lr=0.05)
    path = tmp_path / "deep.model"
    m.save(str(path))
    loaded = gpc.models.DeepGP.load(str(path))
    assert loaded.num_layers == 2
    assert loaded.drop_residual is True
    # Mean-path predict is deterministic; ELBO is MC so seed/RNG can differ.
    mu0, _ = m.predict(X)
    mu1, _ = loaded.predict(X)
    assert np.allclose(mu0, mu1, atol=1e-6)


def test_residual_off_and_str():
    X, Y = _sinc_toy(n=12)
    m = gpc.models.DeepGP(
        X, Y, layers=2, num_inducing=4, drop_residual=True, chi=0.5
    )
    assert m.drop_residual is True
    text = str(m)
    assert "Deep GP" in text
    assert "deepgp" in m.approx
