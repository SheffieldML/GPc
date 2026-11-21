"""CIP-0018 / CIP-001D follow-on: MRD Python façade smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _two_views(n: int = 16, q: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, q))
    # Shared dim 0, private-ish dims 1 and 2.
    Y0 = np.column_stack(
        [
            X[:, 0] + 0.3 * X[:, 1],
            0.7 * X[:, 0] + 0.2 * X[:, 1],
        ]
    ) + 0.05 * rng.normal(size=(n, 2))
    Y1 = np.column_stack(
        [
            X[:, 0] + 0.4 * X[:, 2],
            0.5 * X[:, 0],
            0.2 * X[:, 0] + X[:, 2],
        ]
    ) + 0.05 * rng.normal(size=(n, 3))
    return [Y0, Y1]


def test_mrd_construct_optimize_factorize():
    Ys = _two_views()
    m = gpc.models.MRD(Ys, input_dim=3, num_inducing=6)
    assert m.n == 16
    assert m.num_views == 2
    assert m.input_dim == 3
    assert m.num_inducing == 6
    assert m.output_dim(0) == 2
    assert m.output_dim(1) == 3
    assert "rbfard" in m.kernel_type
    assert m.latent_mean().shape == (16, 3)
    assert m.latent_variance().shape == (16, 3)
    assert m.inducing_inputs().shape == (6, 3)
    before = m.objective()
    m.optimize(iters=25)
    assert m.objective() <= before + 1e-4
    assert m.elbo() == pytest.approx(-m.objective())
    scales0 = m.input_scales(0)
    scales1 = m.input_scales(1)
    assert scales0.shape == (3,)
    assert scales1.shape == (3,)
    fac = m.factorize_space(threshold=0.05)
    assert set(fac.keys()) == {"shared", "private0", "private1"}
    indexed = set(fac["shared"]) | set(fac["private0"]) | set(fac["private1"])
    assert indexed.issubset({0, 1, 2})


def test_mrd_predict_views():
    Ys = _two_views(n=12)
    m = gpc.models.MRD(Ys, input_dim=2, num_inducing=5)
    m.optimize(iters=10)
    X = m.latent_mean()[:3]
    mu0, var0 = m.predict(0, X)
    mu1, var1 = m.predict(1, X)
    assert mu0.shape == (3, 2)
    assert mu1.shape == (3, 3)
    assert np.all(var0 > 0.0) and np.all(var1 > 0.0)
    mu_f, var_f = m.predict_noiseless(0, X)
    assert np.all(var0 >= var_f - 1e-12)


def test_mrd_requires_two_views():
    Ys = _two_views()
    with pytest.raises(gpc.GpcError):
        gpc.models.MRD([Ys[0]], input_dim=2, num_inducing=4)


def test_mrd_mismatched_n():
    Ys = _two_views(n=12)
    bad = [Ys[0], Ys[1][:10]]
    with pytest.raises(gpc.GpcError):
        gpc.models.MRD(bad, input_dim=2, num_inducing=4)
