#!/usr/bin/env python3
"""CIP-003A: print SciPy / GPy reference numbers for testSkewNoise.cpp."""
from __future__ import annotations

import math

import numpy as np
from scipy import special, stats


def inv_mills(z: float) -> float:
    z = float(z)
    return float(
        np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - special.log_ndtr(z))
    )


def d_inv(z: float, lam: float | None = None) -> float:
    if lam is None:
        lam = inv_mills(z)
    return float(-lam * (z + lam))


def d2_inv(z: float, lam: float | None = None) -> float:
    if lam is None:
        lam = inv_mills(z)
    return float(lam * ((z + lam) * (z + 2.0 * lam) - 1.0))


def main() -> None:
    print("# inverse Mills")
    for z in (-3.0, -1.0, 0.0, 0.5, 2.0):
        lam = inv_mills(z)
        print(f"{z:+.1f}  {lam:.16e}  {d_inv(z, lam):.16e}  {d2_inv(z, lam):.16e}")

    y, f, s2, a = 1.2, 0.3, 0.25, 2.0
    sigma = math.sqrt(s2)
    print("# SkewGaussian logpdf", float(stats.skewnorm.logpdf(y, a, loc=f, scale=sigma)))
    print("# Gaussian / alpha=0", float(stats.norm.logpdf(y, loc=f, scale=sigma)))
    e = y - f
    rate, alpha = 1.5, 1.0
    print(
        "# SkewExponential logpdf",
        math.log(rate) - rate * abs(e) + float(special.log_ndtr(alpha * e)),
    )


if __name__ == "__main__":
    main()
