"""CIP-003A: gpc.likelihoods façade vs SciPy / closed-form references."""

from __future__ import annotations

import math

import numpy as np
import pytest

import gpc


def test_likelihood_types_exported():
    names = gpc._gpc.likelihood_types()
    assert "skew_gaussian" in names
    assert "student_t" in names
    assert "poisson" in names


def test_skew_gaussian_logpdf_matches_scipy():
    scipy = pytest.importorskip("scipy")
    from scipy import stats

    y = np.array([1.2])
    f = np.array([0.3])
    lik = gpc.likelihoods.SkewGaussian(variance=0.25, alpha=2.0)
    got = lik.logpdf(f, y)
    want = stats.skewnorm.logpdf(y, 2.0, loc=f, scale=0.5)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_skew_gaussian_alpha0_matches_gaussian():
    y = np.array([1.2, -0.5])
    f = np.array([0.3, 0.1])
    skew = gpc.likelihoods.SkewGaussian(variance=0.25, alpha=0.0)
    gauss = gpc.likelihoods.Gaussian(variance=0.25)
    np.testing.assert_allclose(skew.logpdf(f, y), gauss.logpdf(f, y), rtol=1e-12)


def test_poisson_logpdf():
    y = np.array([3.0])
    f = np.array([0.5])
    lik = gpc.likelihoods.Poisson()
    got = float(lik.logpdf(f, y)[0])
    rate = math.exp(0.5)
    want = -rate + 3.0 * 0.5 - math.lgamma(4.0)
    assert abs(got - want) < 1e-12


def test_student_t_and_from_dict():
    lik = gpc.likelihoods.from_dict(
        {"type": "student_t", "variance": 0.5, "deg_free": 5.0}
    )
    assert isinstance(lik, gpc.likelihoods.StudentT)
    y = np.array([1.2])
    f = np.array([0.3])
    got = float(lik.logpdf(f, y)[0])
    assert abs(got - (-1.4640183713192005)) < 1e-10


def test_binomial_n1_matches_bernoulli():
    f = np.array([0.2])
    binom = gpc.likelihoods.Binomial(num_trials=1)
    bern = gpc.likelihoods.Bernoulli()
    np.testing.assert_allclose(
        binom.logpdf(f, np.array([1.0])), bern.logpdf(f, np.array([1.0])), rtol=1e-12
    )
