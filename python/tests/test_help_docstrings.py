"""CIP-002F / REQ-0017: public gpc surface has help()-visible docstrings."""

from __future__ import annotations

import numpy as np
import pytest

import gpc


def _nonempty(doc: str | None) -> bool:
    return isinstance(doc, str) and len(doc.strip()) > 0


def test_package_and_factory_docstrings():
    assert _nonempty(gpc.__doc__)
    assert _nonempty(gpc.GPRegression.__doc__)
    assert _nonempty(gpc.SparseGPRegression.__doc__)
    assert _nonempty(gpc.models.GPRegression.__doc__)
    assert _nonempty(gpc.models.SparseGPRegression.__doc__)
    assert _nonempty(gpc.kern.__doc__)


@pytest.mark.parametrize(
    "cls",
    [
        gpc.kern.RBF,
        gpc.kern.ExpQuad,
        gpc.kern.White,
        gpc.kern.Bias,
        gpc.kern.MLP,
        gpc.kern.ArcCosine,
        gpc.kern.Add,
        gpc.kern.Kern,
    ],
)
def test_kern_class_docstrings(cls):
    assert _nonempty(cls.__doc__)
    assert _nonempty(cls.__init__.__doc__)


def test_bound_model_method_docstrings():
    X = np.linspace(0.0, 1.0, 8)[:, None]
    Y = np.sin(2.0 * np.pi * X)
    m = gpc.GPRegression(X, Y)
    assert _nonempty(type(m).__doc__)
    for name in (
        "set_by_path",
        "get_by_path",
        "param_paths",
        "optimize",
        "objective",
        "predict",
    ):
        assert _nonempty(getattr(m, name).__doc__), name

    Z = np.linspace(0.1, 0.9, 3)[:, None]
    s = gpc.SparseGPRegression(X, Y, Z)
    assert _nonempty(type(s).__doc__)
    for name in ("optimize", "predict", "inducing_inputs", "param_paths"):
        assert _nonempty(getattr(s, name).__doc__), name
