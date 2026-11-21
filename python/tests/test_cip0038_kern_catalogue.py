"""CIP-0038: GPy-named kernels on gpc.kern + path segments."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


@pytest.fixture
def tiny_xy():
    X = np.linspace(0.1, 1.0, 6)[:, None]
    Y = np.sin(2.0 * np.pi * X)
    return X, Y


def test_phase_a_constructors_and_paths(tiny_xy):
    X, Y = tiny_xy
    cases = [
        (gpc.kern.Matern32(1, lengthscale=0.7), "Mat32.lengthscale", 0.7),
        (gpc.kern.Matern52(1, lengthscale=1.2), "Mat52.lengthscale", 1.2),
        (gpc.kern.Exponential(1, lengthscale=0.5), "Exponential.lengthscale", 0.5),
        (gpc.kern.OU(1, lengthscale=0.4), "Exponential.lengthscale", 0.4),
        (gpc.kern.RatQuad(1, lengthscale=0.8, power=1.5), "RatQuad.power", 1.5),
        (gpc.kern.StdPeriodic(1, period=2.0, lengthscale=0.9), "std_periodic.period", 2.0),
        (gpc.kern.Cosine(1, lengthscale=1.1), "Cosine.lengthscale", 1.1),
        (gpc.kern.Linear(1, variance=0.3), "linear.variance", 0.3),
        (gpc.kern.Poly(1, degree=3, variance=0.5), "poly.variance", 0.5),
        (gpc.kern.Brownian(1, variance=0.2), "Brownian.variance", 0.2),
    ]
    for k, path, want in cases:
        m = gpc.GPRegression(X, Y, kernel=k + gpc.kern.White(1, variance=0.05))
        got = m.get_by_path(path)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6), path


def test_phase_b_sinc_expquadcosine(tiny_xy):
    X, Y = tiny_xy
    m1 = gpc.GPRegression(
        X, Y, kernel=gpc.kern.Sinc(1, lengthscale=0.6) + gpc.kern.White(1, variance=0.05)
    )
    assert m1.get_by_path("Sinc.lengthscale") == pytest.approx(0.6)
    m2 = gpc.GPRegression(
        X,
        Y,
        kernel=gpc.kern.ExpQuadCosine(1, lengthscale=0.7, period=1.5)
        + gpc.kern.White(1, variance=0.05),
    )
    assert m2.get_by_path("ExpQuadCosine.period") == pytest.approx(1.5)


def test_compound_matern_periodic(tiny_xy):
    X, Y = tiny_xy
    k = gpc.kern.Matern32(1) + gpc.kern.StdPeriodic(1, period=2.0) + gpc.kern.White(1)
    m = gpc.GPRegression(X, Y, kernel=k)
    paths = m.param_paths()
    assert any("Mat32." in p for p in paths)
    assert any("std_periodic." in p for p in paths)


def test_ard_rbf_and_matern(tiny_xy):
    X = np.random.default_rng(0).normal(size=(8, 2))
    Y = X[:, :1]
    m = gpc.GPRegression(
        X, Y, kernel=gpc.kern.Matern32(2, ARD=True) + gpc.kern.White(2, variance=0.1)
    )
    paths = m.param_paths()
    assert any("Mat32." in p for p in paths)
    assert any("inputScale_" in p for p in paths)


def test_brownian_rejects_2d():
    with pytest.raises(ValueError, match="input_dim"):
        gpc.kern.Brownian(2)


def test_product_rbf_linear(tiny_xy):
    X, Y = tiny_xy
    k = gpc.kern.RBF(1, lengthscale=0.8, variance=1.2) * gpc.kern.Linear(
        1, variance=0.4
    )
    assert isinstance(k, gpc.kern.Prod)
    parts = k.to_parts()
    assert len(parts) == 1
    assert parts[0]["type"] == "tensor"
    assert [p["type"] for p in parts[0]["parts"]] == ["rbf", "lin"]

    m = gpc.GPRegression(X, Y, kernel=k + gpc.kern.White(1, variance=0.05))
    paths = m.param_paths()
    assert any("tensor.rbf." in p for p in paths)
    assert any("tensor.linear." in p for p in paths)
    assert m.get_by_path("tensor.rbf.lengthscale") == pytest.approx(0.8)
    assert m.get_by_path("tensor.linear.variance") == pytest.approx(0.4)


def test_product_rejects_white_inside():
    X = np.linspace(0.1, 1.0, 6)[:, None]
    Y = np.sin(2.0 * np.pi * X)
    k = gpc.kern.RBF(1) * gpc.kern.White(1, variance=0.1)
    with pytest.raises(gpc.GpcError, match="white"):
        gpc.GPRegression(X, Y, kernel=k)
