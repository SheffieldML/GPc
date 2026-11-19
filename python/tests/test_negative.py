"""CIP-001D Phase D: loud-error tests for bad shapes / missing paths."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def test_unknown_path_raises():
    X = np.linspace(0.0, 1.0, 8)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.1
    m = gpc.GPRegression(X, Y)
    with pytest.raises(gpc.GpcError, match="Unknown"):
        m.set_by_path("no.such.path", 1.0)
    with pytest.raises(gpc.GpcError, match="Unknown"):
        m.get_by_path("rbf.not_a_param")


def test_predict_dim_mismatch():
    X = np.linspace(0.0, 1.0, 8)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.1
    m = gpc.GPRegression(X, Y)
    with pytest.raises(gpc.GpcError, match="dimension"):
        m.predict(np.zeros((3, 2)))
