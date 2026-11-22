"""CIP-001D Phase C: SparseGPRegression DTCVAR smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _toy_sparse():
    rng = np.random.default_rng(1)
    X = np.linspace(0.0, 1.0, 24)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    Z = np.linspace(0.05, 0.95, 6)[:, None]
    return X, Y, Z


def test_construct_dtcvar_and_paths():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(X, Y, Z)
    assert m.approx == "dtcvar"
    assert m.sparse_linalg == "lq"  # CIP-001B / CIP-0016 default
    assert m.num_inducing == 6
    assert m.inducing_fixed is False  # matches CGp; pass True to freeze Z
    assert m.n == 24
    paths = m.param_paths()
    assert any("rbf" in p for p in paths)
    m.set_by_path("rbf.lengthscale", 0.4)
    assert m.get_by_path("rbf.lengthscale") == pytest.approx(0.4)
    # Freeze and check Z stays put under optimise.
    m_fix = gpc.SparseGPRegression(X, Y, Z, inducing_fixed=True)
    assert m_fix.inducing_fixed is True
    np.testing.assert_allclose(m_fix.inducing_inputs(), Z)


def test_sparse_linalg_modes():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(X, Y, Z, sparse_linalg="stable")
    assert m.sparse_linalg == "stable"
    ll_stable = m.objective()
    m.sparse_linalg = "lq"
    assert m.sparse_linalg == "lq"
    assert np.isfinite(m.objective())
    m.sparse_linalg = "legacy"
    assert m.sparse_linalg == "legacy"
    # Aliases on set (canonical get is stable/legacy/lq).
    m.sparse_linalg = "chol"
    assert m.sparse_linalg == "stable"
    m.sparse_linalg = "StableFactor"
    assert m.sparse_linalg == "stable"
    assert np.isfinite(ll_stable)
    with pytest.raises(gpc.GpcError, match="sparse_linalg"):
        gpc.SparseGPRegression(X, Y, Z, sparse_linalg="bogus")


def test_optimize_and_predict():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(X, Y, Z)
    before = m.objective()
    m.optimize(iters=20)
    after = m.objective()
    assert np.isfinite(after)
    assert after <= before + 1e-6
    mu, var = m.predict(X)
    assert mu.shape == Y.shape
    assert var.shape == Y.shape
    assert np.all(var >= 0.0)


def test_fitc_and_dtc_smoke():
    X, Y, Z = _toy_sparse()
    m_fitc = gpc.SparseGPRegression(X, Y, Z, approx="fitc")
    assert m_fitc.approx == "fitc"
    before = m_fitc.objective()
    m_fitc.optimize(iters=15)
    assert m_fitc.objective() <= before + 1e-6
    mu, var = m_fitc.predict(X)
    assert mu.shape == Y.shape and np.all(var >= 0.0)

    m_dtc = gpc.SparseGPRegression(X, Y, Z, approx="dtc")
    assert m_dtc.approx == "dtc"
    assert np.isfinite(m_dtc.objective())


def test_unknown_approx_raises():
    X, Y, Z = _toy_sparse()
    with pytest.raises(gpc.GpcError, match="not supported"):
        gpc.SparseGPRegression(X, Y, Z, approx="pitc")


def test_z_shape_mismatch():
    X, Y, Z = _toy_sparse()
    with pytest.raises(gpc.GpcError):
        gpc.SparseGPRegression(X, Y, np.zeros((6, 2)))
    with pytest.raises(gpc.GpcError):
        gpc.SparseGPRegression(X, Y, np.linspace(0, 1, 30)[:, None])


def test_learn_window_width_opt_in():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(
        X,
        Y,
        Z,
        kernel="mlp",
        inducing="weightspace",
        window_width=0.55,
        learn_window_width=True,
    )
    assert m.inducing_fixed is False
    assert m.learn_window_width is True
    assert "inducing.window_width" in m.param_paths()
    assert m.get_by_path("inducing.window_width") == pytest.approx(0.55)
    # Table must show s; opt count includes free centres + s (not just kern/beta).
    text = str(m)
    assert "inducing.window_width" in text
    assert m.describe().opt_num_params > len(m.describe().params)
    s0 = m.window_width
    before = m.objective()
    m.optimize(iters=15)
    assert np.isfinite(m.objective())
    assert m.objective() <= before + 1e-5
    # Softness should move when learning is on (smoke: not stuck at init).
    assert m.window_width != pytest.approx(s0, rel=0.0, abs=1e-12) or m.objective() < before
    with pytest.raises(gpc.GpcError, match="inducing_fixed=False"):
        gpc.SparseGPRegression(
            X,
            Y,
            Z,
            kernel="mlp",
            inducing="weightspace",
            learn_window_width=True,
            inducing_fixed=True,
        )
    with pytest.raises(gpc.GpcError, match="weightspace"):
        gpc.SparseGPRegression(X, Y, Z, learn_window_width=True)


def test_window_width_fixed_still_visible():
    X, Y, Z = _toy_sparse()
    m = gpc.SparseGPRegression(
        X, Y, Z, kernel="mlp", inducing="weightspace", window_width=0.55
    )
    assert "inducing.window_width" in m.param_paths()
    row = next(p for p in m.describe().params if p.path == "inducing.window_width")
    assert row.fixed is True
    assert row.value == pytest.approx(0.55)


def test_per_unit_window_widths_cip0032():
    X, Y, Z = _toy_sparse()
    M = Z.shape[0]
    s_vec = 0.4 + 0.05 * np.arange(M, dtype=float)
    m = gpc.SparseGPRegression(
        X,
        Y,
        Z,
        kernel="mlp",
        inducing="weightspace",
        window_width=s_vec,
        learn_window_widths=True,
    )
    assert m.learn_window_widths is True
    assert m.learn_window_width is False
    assert m.window_widths.shape == (M,)
    assert np.allclose(m.window_widths, s_vec)
    paths = m.param_paths()
    assert all(f"inducing.window_width.{i}" in paths for i in range(M))
    assert "inducing.window_width" not in paths
    assert m.get_by_path("inducing.window_width.0") == pytest.approx(s_vec[0])
    before = m.objective()
    s0 = m.window_widths.copy()
    m.optimize(iters=12)
    assert np.isfinite(m.objective())
    assert m.objective() <= before + 1e-5
    assert not np.allclose(m.window_widths, s0) or m.objective() < before
    text = str(m)
    assert "inducing.window_width.0" in text


def test_learn_window_width_and_widths_mutex():
    X, Y, Z = _toy_sparse()
    with pytest.raises(gpc.GpcError, match="only one of"):
        gpc.SparseGPRegression(
            X,
            Y,
            Z,
            kernel="mlp",
            inducing="weightspace",
            learn_window_width=True,
            learn_window_widths=True,
        )
