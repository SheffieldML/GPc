#!/usr/bin/env python3
"""CIP-003E: K-class deep GP on a spiral toy, three likelihoods side by side.

Compares a Gaussian regression of centred one-hot targets (the pre-CIP-003E baseline,
with class probabilities by Monte Carlo over the per-class Gaussians) with the
``robustmax`` and ``softmax`` likelihoods, which give class probabilities directly.

Run from the repo after ``make -C python``::

    PYTHONPATH=python python/.venv/bin/python python/examples/deepgp_multiclass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gpc

NUM_CLASSES = 4


def spirals(n_per_class: int, rng: np.random.Generator, noise: float = 0.9):
    """K interleaved spiral arms in the plane; labels are integers 0..K-1."""
    X, y = [], []
    for k in range(NUM_CLASSES):
        r = np.linspace(0.2, 1.0, n_per_class)
        theta = 2.0 * np.pi * k / NUM_CLASSES + 2.5 * r
        pts = np.c_[r * np.cos(theta), r * np.sin(theta)]
        X.append(pts + noise * 0.3 * rng.standard_normal(pts.shape))
        y.append(np.full(n_per_class, k))
    X, y = np.vstack(X), np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def metrics(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    prob = np.clip(prob, 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    acc = float(np.mean(np.argmax(prob, axis=1) == y))
    nll = float(-np.mean(np.log(prob[np.arange(len(y)), y])))
    return acc, nll


def gaussian_baseline_proba(mu, var, rng, draws: int = 2000) -> np.ndarray:
    """P(class k) = P(f_k is the largest) under independent N(mu_k, var_k)."""
    f = mu[None] + np.sqrt(var)[None] * rng.standard_normal((draws,) + mu.shape)
    win = np.argmax(f, axis=2)
    return np.stack([(win == k).mean(axis=0) for k in range(mu.shape[1])], axis=1)


CONFIGS = [
    ("gaussian", "gaussian", {}),
    ("robustmax e=.01", "robustmax", {"eps": 1e-2}),
    ("robustmax e=.10", "robustmax", {"eps": 0.10}),
    ("robustmax e=.25", "robustmax", {"eps": 0.25}),
    ("softmax", "softmax", {}),
]


def fit(X, y, kind, opts, iters, seed):
    kw = dict(layers=2, hidden=2, num_inducing=30, kernel="rbf")
    if kind == "gaussian":
        Y = 2.0 * np.eye(NUM_CLASSES)[y] - 1.0
        m = gpc.models.DeepGP(X, Y, beta=4.0, **kw)
    elif kind == "robustmax":
        m = gpc.models.DeepGP(X, y, likelihood=gpc.likelihoods.RobustMax(NUM_CLASSES, **opts), **kw)
    else:
        m = gpc.models.DeepGP(X, y, likelihood=gpc.likelihoods.Softmax(NUM_CLASSES, num_samples=128), **kw)
    # Adam only: CVI (natgrad_gamma) is not supported for robust-max (see CIP-003E).
    m.optimize(iters=iters, adam_lr=0.03, seed=seed)
    return m


def main(iters: int = 600, repeats: int = 3) -> None:
    Xte, yte = spirals(250, np.random.default_rng(1000))
    acc = {name: [] for name, _, _ in CONFIGS}
    nll = {name: [] for name, _, _ in CONFIGS}
    for r in range(repeats):
        Xtr, ytr = spirals(60, np.random.default_rng(r))
        for name, kind, opts in CONFIGS:
            m = fit(Xtr, ytr, kind, opts, iters, seed=3 + r)
            if kind == "gaussian":
                mu, var = m.predict(Xte)
                p = gaussian_baseline_proba(mu, var, np.random.default_rng(5))
            else:
                p = m.predict_proba(Xte)
            a, n = metrics(p, yte)
            acc[name].append(a)
            nll[name].append(n)
    print(f"{repeats} training sets of {NUM_CLASSES * 60} points; test set of {NUM_CLASSES * 250}")
    print(f"{'likelihood':<18}{'test acc':>16}{'test NLL':>16}")
    for name, _, _ in CONFIGS:
        print(f"{name:<18}{np.mean(acc[name]):>10.3f} +-{np.std(acc[name]):.3f}"
              f"{np.mean(nll[name]):>10.3f} +-{np.std(nll[name]):.3f}")
    for name in ("softmax", "robustmax e=.10"):
        assert np.mean(acc[name]) > 0.6, f"{name} test accuracy too low"
        assert np.isfinite(np.mean(nll[name]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
