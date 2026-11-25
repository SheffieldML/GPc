"""CIP-003B / REQ-001C: HMC over GP hyperparameters from Python."""

from __future__ import annotations

import math

import numpy as np
import pytest

import gpc


def _data(n=25, seed=0):
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 6, n)[:, None]
    Y = np.sin(X) + 0.1 * rng.standard_normal((n, 1))
    return X, Y


def _model():
    X, Y = _data()
    return gpc.models.GPRegression(X, Y, kernel=gpc.kern.RBF(1))


def _prior_all(m, width_path="rbf.lengthscale"):
    for path in (width_path, "rbf.variance", "likelihood.variance"):
        gpc.inference.set_prior(m, path, "gamma", 2.0, 1.0)


def test_improper_target_is_refused_naming_parameters():
    m = _model()
    with pytest.raises(gpc.GpcError, match="improper"):
        gpc.inference.hmc(m, num_samples=5, burn_in=0)


def test_allow_improper_overrides():
    m = _model()
    res = gpc.inference.hmc(m, num_samples=5, burn_in=0, allow_improper=True)
    assert res.samples.shape == (5, 3)


def test_samples_shape_names_positivity_and_acceptance():
    m = _model()
    _prior_all(m)
    res = gpc.inference.hmc(m, num_samples=400, burn_in=150, seed=3)
    assert set(res.names) == {
        "rbf.variance",
        "rbf.lengthscale",
        "likelihood.variance",
    }
    assert res.samples.shape == (400, 3)
    assert res.log_posterior.shape == (400,)
    assert np.all(res.samples > 0.0)
    assert res.accept_rate > 0.5
    assert np.all(np.isfinite(res.log_posterior))


def test_model_is_restored_after_sampling():
    m = _model()
    _prior_all(m)
    before = {p: m.get_by_path(p) for p in m.param_paths()}
    gpc.inference.hmc(m, num_samples=50, burn_in=20, fix=["rbf.variance"])
    after = {p: m.get_by_path(p) for p in m.param_paths()}
    assert before == after
    # The fix was temporary: three parameters sample again.
    res = gpc.inference.hmc(m, num_samples=5, burn_in=0)
    assert res.samples.shape[1] == 3


def test_fix_removes_a_column():
    m = _model()
    _prior_all(m)
    res = gpc.inference.hmc(m, num_samples=20, burn_in=5, fix=["rbf.variance"])
    assert "rbf.variance" not in res.names
    assert res.samples.shape == (20, 2)


def test_same_seed_reproduces():
    a = _model()
    _prior_all(a)
    b = _model()
    _prior_all(b)
    ra = gpc.inference.hmc(a, num_samples=30, burn_in=10, seed=5)
    rb = gpc.inference.hmc(b, num_samples=30, burn_in=10, seed=5)
    np.testing.assert_allclose(ra.samples, rb.samples)


def test_csv_round_trip(tmp_path):
    m = _model()
    _prior_all(m)
    res = gpc.inference.hmc(m, num_samples=20, burn_in=5)
    path = tmp_path / "s.csv"
    res.to_csv(str(path))
    header = path.read_text().splitlines()[0].split(",")
    assert header == ["iteration", "log_posterior", *res.names]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 2:], res.samples, rtol=1e-14)
    np.testing.assert_allclose(data[:, 1], res.log_posterior, rtol=1e-14)


@pytest.mark.parametrize("width_path", ["rbf.lengthscale", "rbf.inv_lengthscale"])
def test_length_scale_posterior_matches_grid(width_path):
    """Samples are reported as lengthscales whichever name carries the prior.

    The stored width is gamma = l**-2 and exp(-objective) is the density in
    gamma, so the density of l on a grid is exp(-objective(gamma(l))) |dgamma/dl|.
    """
    m = _model()
    _prior_all(m, width_path)
    m.set_by_path("rbf.variance", 1.0)
    m.set_by_path("likelihood.variance", 0.01)
    res = gpc.inference.hmc(
        m,
        num_samples=4000,
        burn_in=400,
        seed=11,
        fix=["rbf.variance", "likelihood.variance"],
    )
    ls = res.samples[:, res.names.index("rbf.lengthscale")]
    grid = np.linspace(0.3, 15.0, 6000)
    logp = np.empty_like(grid)
    for i, l in enumerate(grid):
        m.set_by_path("rbf.lengthscale", l)
        logp[i] = -m.objective() + np.log(2.0) - 3.0 * np.log(l)
    p = np.exp(logp - logp.max())
    p /= np.trapezoid(p, grid)
    gm = np.trapezoid(grid * p, grid)
    # Standard error from non-overlapping batch means (accounts for correlation).
    batches = ls.reshape(40, -1).mean(axis=1)
    se = batches.std(ddof=1) / np.sqrt(len(batches))
    assert abs(ls.mean() - gm) < 4.0 * se + 1e-3, (ls.mean(), gm, se)


def test_sparse_model_samples():
    X, Y = _data()
    Z = np.linspace(0, 6, 6)[:, None]
    m = gpc.models.SparseGPRegression(
        X, Y, Z, kernel=gpc.kern.RBF(1), approx="fitc", inducing_fixed=True
    )
    _prior_all(m)
    res = gpc.inference.hmc(m, num_samples=100, burn_in=50)
    assert res.samples.shape[0] == 100
    assert np.all(res.samples > 0.0)
    assert "likelihood.variance" in res.names


def test_lengthscale_prior_is_stated_on_the_lengthscale():
    """A gamma prior on rbf.lengthscale differs from one on the stored width."""
    m = _model()
    m.set_by_path("rbf.lengthscale", 2.0)
    gpc.inference.set_prior(m, "rbf.lengthscale", "gamma", 3.0, 2.0)
    with_prior = m.objective()
    m2 = _model()
    m2.set_by_path("rbf.lengthscale", 2.0)
    base = m2.objective()
    # The prior on gamma is q(l) |dl/dgamma| with |dl/dgamma| = 0.5 l**3.
    a, b, l = 3.0, 2.0, 2.0
    log_q = a * np.log(b) - math.lgamma(a) + (a - 1) * np.log(l) - b * l
    expected = -(log_q + np.log(0.5 * l**3))
    assert abs((with_prior - base) - expected) < 1e-8, (with_prior - base, expected)


def test_prior_spec_validation():
    assert gpc.inference.prior_spec("gamma", 2, 1) == "gamma:2.0,1.0"
    with pytest.raises(ValueError):
        gpc.inference.prior_spec("cauchy", 1)
    with pytest.raises(ValueError):
        gpc.inference.prior_spec("gamma", 1)
