#!/usr/bin/env python3
"""Generate matio fixtures for CIP-000D coregionalize / ICM / indexwhite.

Prefers GPy when importable; otherwise uses the same B = WW' + diag(kappa)
and ICM product formulae. Params are in *untransformed* space for the
standalone testCoregionalize path; a future testKern hook can consume these.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "matfiles"
RNG = np.random.default_rng(20261116)


def try_gpy():
    try:
        import GPy
        from GPy.util.multioutput import ICM, build_XY

        return GPy, ICM, build_XY
    except Exception:
        return None, None, None


def B_of(W, kappa):
    return W @ W.T + np.diag(kappa)


def main():
    OUT.mkdir(exist_ok=True)
    D, R = 2, 1
    W = np.array([[1.0], [0.5]])
    kappa = np.array([0.1, 0.2])
    B = B_of(W, kappa)

    # Stacked design: two outputs, unequal N, index in last column.
    X1 = RNG.normal(size=(3, 1))
    X2 = RNG.normal(size=(2, 1))
    X1[1, 0] = X1[0, 0]  # near-duplicate spatial point across outputs
    Y1 = RNG.normal(size=(3, 1))
    Y2 = RNG.normal(size=(2, 1))

    GPy, ICM, build_XY = try_gpy()
    if build_XY is not None:
        X, Y, _ = build_XY([X1, X2], [Y1, Y2])
        kspace = GPy.kern.RBF(1, variance=1.0, lengthscale=1.0)
        # GPy lengthscale <-> inverseWidth: iw = 1/ls^2
        icm = ICM(1, 2, kspace, W_rank=1, W=W.copy(), kappa=kappa.copy())
        K = icm.K(X)
        source = "gpy"
    else:
        idx = np.concatenate([np.zeros(3), np.ones(2)])[:, None]
        X = np.hstack([np.vstack([X1, X2]), idx])
        Y = np.vstack([Y1, Y2])
        Xs = X[:, :1]
        r2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        Ks = np.exp(-0.5 * r2)  # inverseWidth=1
        ii = X[:, 1].astype(int)
        K = Ks * B[np.ix_(ii, ii)]
        source = "numpy"

    sig = np.array([0.3, 0.7])
    diag_noise = sig[X[:, 1].astype(int)]

    savemat(
        OUT / "coregionalizeIcmTest.mat",
        {
            "X": X,
            "Y": Y,
            "W": W,
            "kappa": kappa,
            "B": B,
            "K": K,
            "indexNoise": diag_noise,
            "inverseWidth": 1.0,
            "variance": 1.0,
            "source": source,
        },
    )
    print(f"Wrote {OUT / 'coregionalizeIcmTest.mat'} via {source}")


if __name__ == "__main__":
    main()
