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


def test_str_contains_canonical_paths():
    X, Y = _toy()
    m = gpc.GPRegression(X, Y)
    m.set_by_path("rbf.lengthscale", 0.75)
    text = str(m)
    assert "kern.rbf.lengthscale" in text or "rbf.lengthscale" in text
    desc = m.describe()
    assert desc.title
    assert any("lengthscale" in p.path for p in desc.params)
