"""CIP-003D phase 5: Hermite inducing through the Python façade.

Every case where what is delivered differs from what was asked must either
raise (E1-E4) or warn (W1-W4); a well-conditioned control must stay silent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import gpc
from gpc.models import DeepGP, SparseGPRegression


def _data(n=120, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = np.tanh(X[:, :1]) + 0.1 * rng.normal(size=(n, 1))
    return X, Y


def _build(d, m, **kw):
    X, Y = _data(max(m + 5, 120), d)
    return SparseGPRegression(X, Y, X[:m], kernel="mlp", inducing="hermite", **kw)


def _tags(record):
    return sorted({w.message.args[0].split("(CIP-003D ")[1][:2] for w in record
                   if "CIP-003D" in str(w.message)})


def test_control_is_silent_and_default_is_fixed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning fails the test
        m = _build(2, 20)
    assert m.inducing_type == "hermite"
    assert m.inducing_fixed is True
    assert m.hermite_warnings() == []
    text = str(m)
    assert "Hermite layout: hermite M=20 D'=3" in text
    assert "no free parameters" in text
    assert "captured_variance=" in text


def test_explicit_inducing_fixed_true_is_accepted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = _build(2, 20, inducing_fixed=True)
    assert m.inducing_fixed is True


def test_points_default_is_unchanged():
    X, Y = _data(40, 1)
    m = SparseGPRegression(X, Y, X[:10], kernel="rbf")
    assert m.inducing_fixed is False
    assert m.inducing_type == "points"


def test_E1_capacity_exceeded_names_both_numbers():
    X, Y = _data(400, 2)
    with pytest.raises(gpc.GpcError, match=r"capacity") as ei:
        SparseGPRegression(X, Y, X[:100], kernel="mlp", inducing="hermite")
    msg = str(ei.value)
    assert "63" in msg and "100" in msg


def test_E2_non_mlp_kernel_refused():
    X, Y = _data(60, 2)
    with pytest.raises(gpc.GpcError, match=r"E2"):
        SparseGPRegression(X, Y, X[:10], kernel="rbf", inducing="hermite")
    with pytest.raises(gpc.GpcError, match=r"E2"):
        SparseGPRegression(X, Y, X[:10], kernel="arccos", inducing="hermite")


@pytest.mark.parametrize(
    "kw",
    [
        {"window_width": 0.5},
        {"window_width": np.ones(10)},
        {"learn_window_width": True},
        {"learn_window_widths": True},
    ],
)
def test_E3_width_options_refused(kw):
    X, Y = _data(60, 2)
    with pytest.raises(gpc.GpcError, match=r"E3"):
        SparseGPRegression(X, Y, X[:10], kernel="mlp", inducing="hermite", **kw)


def test_E4_explicit_unfixed_refused():
    X, Y = _data(60, 2)
    with pytest.raises(gpc.GpcError, match=r"E4"):
        SparseGPRegression(
            X, Y, X[:10], kernel="mlp", inducing="hermite", inducing_fixed=False
        )


def test_W1_ill_conditioned_Kuu_warns():
    with pytest.warns(UserWarning, match=r"W1") as rec:
        m = _build(1, 14)
    assert "W1" in _tags(rec)
    assert any("W1" in w for w in m.hermite_warnings())


def test_W2_near_capacity_warns():
    with pytest.warns(UserWarning, match=r"W2") as rec:
        _build(2, 63)
    assert "W2" in _tags(rec)


def test_W3_text_is_wired_through_C_plus_plus_only():
    # The captured fraction of an MLP layout bottoms out near 0.67 (degree-9
    # truncation), so the placeholder threshold of 0.5 cannot be reached from
    # Python. W3 is exercised directly in testInducingHermite; phase 6
    # calibrates the threshold. Here we check the number is reported.
    m = _build(2, 20)
    line = next(h for h in m.describe().header_lines if h.startswith("Hermite layout"))
    frac = float(line.split("captured_variance=")[1].rstrip("%")) / 100.0
    assert 0.5 < frac <= 1.0


def test_deepgp_hermite_layout_and_W4():
    X, Y = _data(80, 2)
    with pytest.warns(UserWarning, match=r"W4") as rec:
        d = DeepGP(X, Y, layers=2, num_inducing=20, kernel="mlp", inducing="hermite")
    assert "W4" in _tags(rec)
    text = str(d)
    assert "layer 0 hermite layout" in text and "layer 1 hermite layout" in text
    assert all("hidden layer 0" in w or "layer" in w for w in d.hermite_warnings())


def test_deepgp_single_layer_has_no_W4():
    X, Y = _data(80, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DeepGP(X, Y, layers=1, num_inducing=20, kernel="mlp", inducing="hermite")


def test_deepgp_E2_E3():
    X, Y = _data(80, 2)
    with pytest.raises(gpc.GpcError, match=r"E2"):
        DeepGP(X, Y, layers=2, num_inducing=20, kernel="rbf", inducing="hermite")
    with pytest.raises(gpc.GpcError, match=r"E3"):
        DeepGP(
            X, Y, layers=2, num_inducing=20, kernel="mlp", inducing="hermite",
            learn_window_width=True,
        )


def test_sparse_hermite_optimises_kernel_params_only():
    m = _build(2, 20)
    before = m.objective()
    m.optimize(iters=15)
    assert m.objective() <= before + 1e-9


def test_layout_is_reproducible_across_constructions():
    a = _build(2, 20)
    # Burn an odd number of normals in between; the layout must not depend on it.
    np.random.default_rng(3).normal(size=3)
    b = _build(2, 20)
    la = next(h for h in a.describe().header_lines if h.startswith("Hermite layout"))
    lb = next(h for h in b.describe().header_lines if h.startswith("Hermite layout"))
    assert la == lb
