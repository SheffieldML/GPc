"""CIP-003E Phase 6: multi-class likelihoods through the Python façade."""

import numpy as np
import pytest

import gpc


def _blobs(n=45, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(n) % 3
    ang = 2.0 * np.pi * k / 3.0
    X = np.c_[np.cos(ang), np.sin(ang)] + 0.15 * rng.standard_normal((n, 2))
    return X, k


def test_from_dict_roundtrip():
    rm = gpc.likelihoods.from_dict(gpc.likelihoods.RobustMax(4, eps=0.02).to_dict())
    assert isinstance(rm, gpc.likelihoods.RobustMax) and rm.num_classes == 4 and rm.eps == 0.02
    sm = gpc.likelihoods.from_dict(gpc.likelihoods.Softmax(3, num_samples=64, seed=7).to_dict())
    assert isinstance(sm, gpc.likelihoods.Softmax) and sm.num_samples == 64 and sm.seed == 7
    assert "robustmax" in gpc._gpc.likelihood_types() and "softmax" in gpc._gpc.likelihood_types()


def test_constructor_validation():
    with pytest.raises(ValueError):
        gpc.likelihoods.RobustMax(1)
    with pytest.raises(ValueError):
        gpc.likelihoods.RobustMax(3, eps=1.5)
    with pytest.raises(ValueError):
        gpc.likelihoods.Softmax(3, num_samples=1)


@pytest.mark.parametrize("lik", [gpc.likelihoods.RobustMax(3, eps=0.05), gpc.likelihoods.Softmax(3, num_samples=128)])
def test_expected_loglik_matches_logpdf_as_variance_vanishes(lik):
    rng = np.random.default_rng(1)
    mu = rng.standard_normal((8, 3)) * 2.0
    y = rng.integers(0, 3, size=8)
    var = np.full_like(mu, 1e-12)
    ell = lik.expected_loglik(y, mu, var)
    assert np.allclose(ell, lik.logpdf(mu, y), atol=1e-6)
    onehot = np.eye(3)[y]
    assert np.allclose(lik.expected_loglik(onehot, mu, var), ell)


def test_predict_proba_is_a_distribution_and_matches_softmax_at_zero_variance():
    rng = np.random.default_rng(2)
    mu = rng.standard_normal((6, 3))
    var = np.full_like(mu, 1e-12)
    for lik in (gpc.likelihoods.RobustMax(3, eps=0.01), gpc.likelihoods.Softmax(3, num_samples=64)):
        p = lik.predict_proba(mu, rng.uniform(0.1, 1.0, mu.shape))
        assert p.shape == (6, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-8) and np.all(p >= 0)
    p = gpc.likelihoods.Softmax(3).predict_proba(mu, var)
    ref = np.exp(mu - mu.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    assert np.allclose(p, ref, atol=1e-8)
    p = gpc.likelihoods.RobustMax(3, eps=0.01).predict_proba(mu, var)
    top = np.argmax(mu, axis=1)
    assert np.allclose(p[np.arange(6), top], 0.99, atol=1e-6)


@pytest.mark.parametrize("name", ["softmax", "robustmax"])
def test_deepgp_integer_labels_train_save_load(tmp_path, name):
    X, lab = _blobs()
    m = gpc.models.DeepGP(X, lab, layers=2, hidden=2, num_inducing=8, likelihood=name)
    assert m.output_dim == 3 and m.likelihood_type == name
    e0 = m.elbo()
    m.optimize(iters=150, adam_lr=0.05, seed=3)
    e1 = m.elbo()
    assert np.isfinite(e1) and e1 > e0
    p = m.predict_proba(X, samples=0)
    assert p.shape == (len(lab), 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert np.mean(np.argmax(p, axis=1) == lab) > 0.6

    path = str(tmp_path / f"{name}.model")
    m.save(path)
    loaded = gpc.models.DeepGP.load(path)
    assert loaded.likelihood_type == name and loaded.output_dim == 3
    # elbo() is a Monte Carlo estimate with an advancing seed; the C++ stream test pins the
    # ELBO under a fixed seed, so here we compare the deterministic mean-path predictive.
    assert np.isfinite(loaded.elbo())
    assert np.allclose(loaded.predict_proba(X), p, atol=1e-10)


def test_deepgp_likelihood_instances_and_one_hot_equivalence():
    X, lab = _blobs(30)
    a = gpc.models.DeepGP(X, lab, hidden=2, num_inducing=6, likelihood=gpc.likelihoods.Softmax(3, num_samples=32, seed=5))
    b = gpc.models.DeepGP(X, np.eye(3)[lab], hidden=2, num_inducing=6, likelihood="softmax")
    assert a.output_dim == b.output_dim == 3
    c = gpc.models.DeepGP(X, lab[:, None], hidden=2, num_inducing=6, likelihood=gpc.likelihoods.RobustMax(3, eps=0.02))
    assert c.likelihood_type == "robustmax" and np.isfinite(c.elbo())


def test_deepgp_label_errors():
    X, lab = _blobs(30)
    with pytest.raises(Exception):
        gpc.models.DeepGP(X, lab + 0.5, likelihood="softmax")
    with pytest.raises(Exception):
        gpc.models.DeepGP(X, lab, likelihood="softmax", num_classes=2)
    with pytest.raises(Exception):
        gpc.models.DeepGP(X, -lab - 1, likelihood="robustmax")
    g = gpc.models.DeepGP(X, X[:, :1], hidden=2, num_inducing=5)
    with pytest.raises(Exception):
        g.predict_proba(X)
    with pytest.raises(TypeError):
        gpc.models.DeepGP(X, lab, likelihood=gpc.likelihoods.Poisson())
