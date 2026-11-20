#!/usr/bin/env python3
"""CIP-002C Phase C: clean-venv checklist after ``pip install gpc``.

Usage (after installing a wheel into a fresh venv)::

    export OPENBLAS_NUM_THREADS=1
    python python/scripts/smoke_installed_gpc.py
"""

from __future__ import annotations

import sys


def main() -> int:
    import numpy as np
    import gpc

    print(f"gpc from {gpc.__file__}")
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 24)[:, None]
    Y = np.sin(2.0 * np.pi * X) + 0.05 * rng.normal(size=X.shape)
    m = gpc.GPRegression(X, Y)
    m.optimize(iters=25)
    mu, var = m.predict(X)
    if mu.shape != Y.shape or var.shape != Y.shape:
        print(f"shape mismatch: mu={mu.shape} var={var.shape} Y={Y.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(mu).all() or not np.isfinite(var).all():
        print("non-finite prediction", file=sys.stderr)
        return 1
    print("smoke_installed_gpc: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
