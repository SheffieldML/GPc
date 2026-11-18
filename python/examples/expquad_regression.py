#!/usr/bin/env python3
"""CIP-0024: GPy ExpQuad → gpc façade sketch (FTC Gaussian regression).

GPy (conceptual)::

    import GPy
    import numpy as np
    X = np.linspace(0, 1, 20)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * np.random.randn(20, 1)
    k = GPy.kern.ExpQuad(1) + GPy.kern.Bias(1) + GPy.kern.White(1)
    m = GPy.models.GPRegression(X, Y, k)
    m.ExpQuad.lengthscale = 0.2
    m.optimize()
    mu, var = m.predict(X)

gpc (this file)::

    same data shape; kernel='eq' selects CEqKern so paths are ExpQuad.*
"""

from __future__ import annotations

import numpy as np

import gpc


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 20)[:, None]
    Y = np.sin(2 * np.pi * X) + 0.05 * rng.normal(size=X.shape)

    m = gpc.GPRegression(X, Y, kernel="eq")
    m.set_by_path("ExpQuad.lengthscale", 0.2)
    m.set_by_path("ExpQuad.variance", 1.0)
    print("paths:", m.param_paths())
    print("objective before:", m.objective())
    m.optimize(iters=50)
    print("objective after:", m.objective())
    mu, var = m.predict(X)
    print("predict shapes:", mu.shape, var.shape, "mean var", float(var.mean()))


if __name__ == "__main__":
    main()
