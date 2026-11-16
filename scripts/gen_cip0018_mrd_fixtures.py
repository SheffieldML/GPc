#!/usr/bin/env python3
"""CIP-0018: dump a tiny two-view GPy MRD ELBO + ARD scales for C++ compare.

Requires GPy. Writes testdata/cip0018/mrd_rbfard_tiny.txt
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "cip0018" / "mrd_rbfard_tiny.txt"


def main() -> None:
    import GPy

    rng = np.random.RandomState(0)
    N, Q, M = 8, 3, 3
    Xtrue = rng.randn(N, Q)
    # Shared dim 0; private-ish 1 vs 2
    Y0 = np.hstack([Xtrue[:, :1], 0.3 * Xtrue[:, 1:2]]) + 0.05 * rng.randn(N, 2)
    Y1 = np.hstack([Xtrue[:, :1], 0.3 * Xtrue[:, 2:3], 0.1 * Xtrue[:, :1]]) + 0.05 * rng.randn(
        N, 3
    )

    m = GPy.models.MRD(
        [Y0, Y1],
        input_dim=Q,
        X=Xtrue.copy(),
        X_variance=0.25 * np.ones((N, Q)),
        num_inducing=M,
        Z=Xtrue[:M].copy(),
        initx="random",
    )
    # Freeze at init for a stable dump (no long optimise).
    elbo = float(m.log_likelihood())
    scales = []
    for i, bg in enumerate(m.bgplvms):
        # RBF ARD lengthscales -> GPc uses inverseWidth + inputScale; dump
        # lengthscales and variance for documentation; C++ gate uses ELBO only.
        k = bg.kern
        scales.append(np.asarray(k.lengthscale).ravel().tolist())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("# CIP-0018 GPy MRD fixture (init, no optimise)\n")
        f.write(f"N {N}\nD0 {Y0.shape[1]}\nD1 {Y1.shape[1]}\nQ {Q}\nM {M}\n")
        f.write(f"elbo {elbo:.17g}\n")
        for i, sc in enumerate(scales):
            f.write(f"lengthscale_{i} {len(sc)}")
            for v in sc:
                f.write(f" {v:.17g}")
            f.write("\n")
        f.write("Y0\n")
        np.savetxt(f, Y0)
        f.write("Y1\n")
        np.savetxt(f, Y1)
    print(f"Wrote {OUT} elbo={elbo}")


if __name__ == "__main__":
    main()
