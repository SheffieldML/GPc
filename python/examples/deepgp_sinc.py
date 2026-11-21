#!/usr/bin/env python3
"""CIP-0035: short DeepGP smoke on a 1-D sinc toy (doctest / example).

Run from the repo after ``make -C python``::

    PYTHONPATH=python python/.venv/bin/python python/examples/deepgp_sinc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gpc


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.linspace(-2.0, 2.0, 40)[:, None]
    Y = np.sin(1.5 * X) + 0.05 * rng.normal(size=X.shape)
    m = gpc.models.DeepGP(
        X, Y, layers=2, hidden=1, num_inducing=8, chi=0.5, beta=5.0
    )
    m.optimize(iters=40, adam_lr=0.05, seed=1)
    mu, var = m.predict(X)
    rmse = float(np.sqrt(np.mean((mu - Y) ** 2)))
    print(f"ELBO={m.elbo():.4f}  RMSE={rmse:.4f}")
    assert np.isfinite(m.elbo())
    assert rmse < 1.0


if __name__ == "__main__":
    main()
