"""CIP-003D phase 6: the CIP-0035 composite task with fixed Hermite windows.

Task generator, depth, hidden width, optimiser settings and the reported
quantities follow ``cip/cip0035/prototype/dgp_train.py`` (three layers, hidden
width 5, chi = 1, Adam 0.01, batch 256, 2500 steps, seed 0). Compare with the
CIP-0035 table:  M=64: 0.546 / 0.79 / -1.01   M=128: 0.315 / 0.21 / -0.63
(test RMSE / test NLL / ELBO per point).

    python deepgp_hermite_composite.py [M ...] [--inducing hermite|points]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from gpc.models import DeepGP


def composite_task(seed: int = 0):
    r = np.random.default_rng(seed)
    Bm = np.linalg.qr(r.normal(size=(10, 2)))[0]
    X = r.normal(size=(2000, 10))
    Xt = r.normal(size=(1000, 10))

    def f(x):
        return np.sin(2 * (x @ Bm)[:, 0] + (x @ Bm)[:, 1] ** 2)

    n = 0.1
    y = f(X) + n * r.normal(size=len(X))
    yt = f(Xt) + n * r.normal(size=len(Xt))
    mu, sd = y.mean(), y.std()
    return X, ((y - mu) / sd)[:, None], Xt, ((yt - mu) / sd)[:, None]


def prototype_equivalent_init(m, layers: int, d_in: int, hidden: int, chi: float = 1.0):
    """Start GPc's MLP kernels where the CIP-0035 prototype starts.

    The prototype kernel is (2/pi) s2 asin(2 xh.xh' / sqrt((1+2|xh|^2)(1+2|xh'|^2))) with
    xh = (h sqrt(w/D), sqrt(b)), w = 1, b = 0.1, s2 = chi/(L-1) hidden and 1 final. That is
    GPc's MLP kernel with weightVariance = 2 w / D, biasVariance = 2 b, variance = (2/pi) s2.
    """
    import math
    for ell in range(layers):
        D = d_in if ell == 0 else hidden
        s2 = 1.0 if ell == layers - 1 else chi / max(layers - 1, 1)
        m.set_layer_param(ell, "weightVariance", 2.0 * 1.0 / D)
        m.set_layer_param(ell, "biasVariance", 2.0 * 0.1)
        m.set_layer_param(ell, "variance", (2.0 / math.pi) * s2)


def run(M: int, inducing: str, seed: int, steps: int, quiet: bool = False, init: str = "default"):
    X, y, Xt, yt = composite_task(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = DeepGP(
            X, y, layers=3, hidden=5, num_inducing=M, kernel="mlp",
            inducing=inducing, chi=1.0, beta=10.0,
        )
    if init == "proto":
        prototype_equivalent_init(m, 3, X.shape[1], 5)
    t0 = time.time()
    m.optimize(iters=steps, batch_size=256, samples=1, adam_lr=0.01, seed=seed)
    sec = time.time() - t0
    mu, var = m.predict(Xt, samples=64, seed=1)
    rmse = float(np.sqrt(np.mean((mu - yt) ** 2)))
    # Gaussian NLL of the sample-averaged moments (the prototype mixes 64 sample densities).
    nll_moment = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (yt - mu) ** 2 / var))
    elbo = float(np.mean([m.elbo() for _ in range(8)])) / len(X)
    if not quiet:
        for w in caught:
            print("  warning:", str(w.message)[:110])
    return dict(M=M, inducing=inducing, seed=seed, init=init, rmse=rmse, nll_moment=nll_moment,
                elbo_per_n=elbo, sec=sec, warnings=[str(w.message) for w in caught])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("M", nargs="*", type=int, default=[64, 128])
    ap.add_argument("--inducing", default="hermite")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--init", choices=["default", "proto"], default="default")
    a = ap.parse_args()
    for M in a.M:
        for s in a.seeds:
            r = run(M, a.inducing, s, a.steps, init=a.init)
            print(f"{a.inducing} init={a.init} M={M} seed={s}: RMSE {r['rmse']:.3f}  NLL(moment) "
                  f"{r['nll_moment']:.3f}  ELBO/N {r['elbo_per_n']:.3f}  [{r['sec']:.0f}s]", flush=True)
