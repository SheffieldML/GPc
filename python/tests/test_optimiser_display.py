"""CIP-001E / CIP-001D B2: live progress callback and structured repr."""

from __future__ import annotations

import numpy as np

import gpc


def _toy():
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 12)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    return X, Y


def test_optimize_callback_outer_iterations():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    events = []

    def cb(e):
        events.append(e)

    m.optimize(iters=15, messages=False, callback=cb)
    assert any(e.phase == gpc.OptimiserPhase.Start for e in events)
    assert any(e.phase == gpc.OptimiserPhase.Iter for e in events)
    assert any(
        e.phase in (gpc.OptimiserPhase.Converged, gpc.OptimiserPhase.MaxIters)
        for e in events
    )
    iters = [e.iter for e in events if e.phase == gpc.OptimiserPhase.Iter]
    assert iters == sorted(iters)
    assert all(np.isfinite(e.objective) for e in events if e.phase == gpc.OptimiserPhase.Iter)


def test_optimize_silent_without_callback():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    events = []
    m.optimize(iters=5, messages=False, callback=None)
    # No callback installed — list stays empty by construction.
    assert events == []


def test_str_paramz_table():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    m.set_by_path("rbf.lengthscale", 0.75)
    text = str(m)
    assert "Name : GP regression" in text
    assert "Objective :" in text
    assert "Number of Parameters :" in text
    assert "Parameters:" in text
    assert "kern.rbf.lengthscale" in text
    assert "Gaussian_noise.variance" in text
    assert "+ve" in text
    desc = m.describe()
    assert any("lengthscale" in p.path for p in desc.params)
    assert any(p.path == "Gaussian_noise.variance" for p in desc.params)


def test_messages_inplace_table_not_cpp_iteration_lines(capsys):
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    m.optimize(iters=8, messages=True)
    out = capsys.readouterr().out
    assert "Iteration:" not in out
    assert "Numerical differences" not in out
    assert "Analytic gradients" not in out
    assert "Initial model:" not in out
    assert "Name : GP regression" in out
    assert "Parameters:" in out
    assert "Running scg" in out
    assert "kern.rbf.lengthscale" in out


def test_messages_and_callback_both_fire():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    events = []
    m.optimize(iters=6, messages=True, callback=events.append)
    assert any(e.phase == gpc.OptimiserPhase.Iter for e in events)


def test_progress_uses_ansi_outside_ipython():
    # Outside IPython there are no widgets / ZMQ display_id — use ANSI rewrite.
    from gpc._display import OptimiserTableProgress, _jupyter_zmq_shell

    assert _jupyter_zmq_shell() is None
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    prog = OptimiserTableProgress(m, max_iters=8)
    assert prog._mode == "ansi"


def test_repr_html_contains_parameter_paths():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    html = m._repr_html_()
    assert "kern.rbf.lengthscale" in html
    assert "<pre" in html
