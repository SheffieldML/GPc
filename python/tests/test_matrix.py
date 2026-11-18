"""CIP-001D Phase A: NumPy ↔ CMatrix round-trip."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def test_from_to_numpy_2d_roundtrip():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    m = gpc.Matrix.from_numpy(x)
    assert m.rows == 2
    assert m.cols == 3
    assert m.get(0, 0) == pytest.approx(1.0)
    assert m.get(1, 2) == pytest.approx(6.0)
    y = m.to_numpy()
    np.testing.assert_allclose(y, x)


def test_from_numpy_1d_becomes_column():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    m = gpc.Matrix.from_numpy(x)
    assert m.rows == 3
    assert m.cols == 1
    y = m.to_numpy()
    np.testing.assert_allclose(y.ravel(), x)


def test_set_get():
    m = gpc.Matrix(2, 2)
    m.set(0, 1, 3.5)
    assert m.get(0, 1) == pytest.approx(3.5)


def test_reject_3d():
    x = np.zeros((2, 2, 2), dtype=np.float64)
    with pytest.raises(gpc.GpcError):
        gpc.Matrix.from_numpy(x)
