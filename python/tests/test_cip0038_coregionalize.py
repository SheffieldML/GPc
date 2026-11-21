"""CIP-0038 Phase D: Coregionalize / ICM / IndexWhite façade."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def test_build_xy_and_stamp():
    X0 = np.linspace(0.0, 1.0, 4)[:, None]
    X1 = np.linspace(0.0, 0.5, 3)[:, None]
    Y0 = np.sin(X0)
    Y1 = np.cos(X1)
    X, Y = gpc.util.multioutput.build_XY([X0, X1], [Y0, Y1])
    assert X.shape == (7, 2)
    assert Y.shape == (7, 1)
    assert np.all(X[:4, 1] == 0.0)
    assert np.all(X[4:, 1] == 1.0)
    Xs = gpc.util.multioutput.stamp_output_index(X[:3].copy(), 1)
    assert np.all(Xs[:, 1] == 1.0)


def test_icm_parts_and_paths():
    X0 = np.linspace(0.1, 2.0, 8)[:, None]
    X1 = np.linspace(0.1, 1.5, 6)[:, None]
    Y0 = np.sin(X0)
    Y1 = np.sin(X1) + 0.5
    icm = gpc.util.multioutput.ICM(1, 2, gpc.kern.RBF(1, lengthscale=0.9))
    assert isinstance(icm, gpc.kern.Prod)
    parts = icm.to_parts()
    assert parts[0]["type"] == "tensor"
    types = [p["type"] for p in parts[0]["parts"]]
    assert types == ["rbf", "coregionalize"]

    m = gpc.GPCoregionalizedRegression([X0, X1], [Y0, Y1], kernel=gpc.kern.RBF(1))
    paths = m.param_paths()
    assert any("tensor.rbf." in p for p in paths)
    assert any("tensor.coregionalize." in p for p in paths)
    assert any("indexwhite.variance_0" in p for p in paths)
    assert m.get_by_path("tensor.rbf.lengthscale") == pytest.approx(1.0)
    mu, _ = m.predict_output(np.linspace(0.1, 2.0, 5)[:, None], output=0)
    assert mu.shape == (5, 1)


def test_lcm_sum_of_icms():
    k = gpc.util.multioutput.LCM(
        1, 2, [gpc.kern.RBF(1), gpc.kern.Linear(1)], W_rank=1
    )
    assert isinstance(k, gpc.kern.Add)
    parts = k.to_parts()
    assert len(parts) == 2
    assert all(p["type"] == "tensor" for p in parts)


def test_indexwhite_rejects_inside_prod():
    X0 = np.linspace(0.1, 1.0, 5)[:, None]
    Y0 = np.sin(X0)
    X, Y = gpc.util.multioutput.build_XY([X0, X0], [Y0, Y0])
    bad = gpc.kern.RBF(1, active_dims=[0]) * gpc.kern.IndexWhite(
        1, output_dim=2, active_dims=[1]
    )
    with pytest.raises(gpc.GpcError, match="indexwhite"):
        gpc.GPRegression(X, Y, kernel=bad)
