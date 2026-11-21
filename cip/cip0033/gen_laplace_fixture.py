#!/usr/bin/env python3
"""CIP-0033 Phase C: numpy twin of CLaplaceInference (GPy Laplace formulas).

Usage:
  python3 cip/cip0033/gen_laplace_fixture.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def ln_cum_gaussian(x: float) -> float:
    return math.log(max(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))), 1e-300))


def grad_ln_cum_gaussian(x: float) -> float:
    phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    Phi = max(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))), 1e-300)
    return phi / Phi


def laplace_derivs(y: float, f: float):
    sign = 1.0 if y > 0.5 else -1.0
    t = sign * f
    r = grad_ln_cum_gaussian(t)
    logp = ln_cum_gaussian(t)
    d1 = sign * r
    W = max(r * (t + r), 1e-12)
    dr = -r * (t + r)
    dWdt = dr * (t + r) + r * (1.0 + dr)
    dWdf = sign * dWdt
    return logp, d1, W, dWdf


def rbf_gram(X, variance=1.0, lengthscale=1.0, jitter=1e-6):
    d2 = (X[:, None, :] - X[None, :, :]) ** 2
    r2 = np.sum(d2, axis=-1) / (lengthscale ** 2)
    return variance * np.exp(-0.5 * r2) + jitter * np.eye(X.shape[0])


def laplace_run(K, Y, max_iters=30, tol=1e-6):
    n = K.shape[0]
    Ki_f = np.zeros(n)
    f = np.zeros(n)
    eps = 1e-12

    def fill(f):
        logp = np.zeros(n)
        d1 = np.zeros(n)
        W = np.zeros(n)
        dW = np.zeros(n)
        for i in range(n):
            logp[i], d1[i], W[i], dW[i] = laplace_derivs(Y[i], f[i])
        return logp, d1, W, dW

    def bstats(W):
        W12 = np.sqrt(np.maximum(W, eps))
        B = np.eye(n) + (W12[:, None] * K) * W12[None, :]
        U = np.linalg.cholesky(B).T
        logDetB = 2.0 * np.sum(np.log(np.diag(U)))
        Xb = np.linalg.solve(U, np.linalg.solve(U.T, np.diag(W12)))
        KWiI = W12[:, None] * Xb
        V = np.linalg.solve(U.T, W12[:, None] * K)
        KiWI = K - V.T @ V
        return KWiI, KiWI, logDetB

    def obj(Ki_f, f, logp):
        return -0.5 * float(Ki_f @ f) + float(np.sum(logp))

    prev = -1e300
    iters = 0
    converged = False
    for it in range(max_iters):
        logp, d1, W, dW = fill(f)
        KWiI, KiWI, logDetB = bstats(W)
        b = W * f + d1
        full = b - KWiI @ (K @ b)
        dKi = full - Ki_f
        old = obj(Ki_f, f, logp)
        step = 1.0
        for _ in range(20):
            Ki_t = Ki_f + step * dKi
            f_t = K @ Ki_t
            logp_t, _, _, _ = fill(f_t)
            new = obj(Ki_t, f_t, logp_t)
            if new >= old - 1e-12:
                break
            step *= 0.5
        else:
            step = max(step, 1e-4)
            Ki_t = Ki_f + step * dKi
            f_t = K @ Ki_t
            logp_t, _, _, _ = fill(f_t)
            new = obj(Ki_t, f_t, logp_t)
        Ki_f, f = Ki_t, f_t
        iters = it + 1
        if it > 0 and abs(new - prev) < tol:
            converged = True
            prev = new
            break
        prev = new

    logp, d1, W, dW = fill(f)
    KWiI, KiWI, logDetB = bstats(W)
    log_marginal = obj(Ki_f, f, logp) - 0.5 * logDetB
    return {
        "log_marginal": log_marginal,
        "converged": converged,
        "iters": iters,
        "f_hat": f,
        "Ki_f": Ki_f,
    }


def main():
    X = np.array([[-1.5], [-0.5], [0.5], [1.5]], dtype=float)
    Y = np.array([1.0, 1.0, 0.0, 0.0])
    K = rbf_gram(X)
    out = laplace_run(K, Y)
    root = Path(__file__).resolve().parent
    fixture = root / "laplace_ftc_fixture.txt"
    lines = [
        "# CIP-0033 FTC Laplace fixture (numpy twin of CLaplaceInference)",
        "N 4",
        "variance 1.0",
        "lengthscale 1.0",
        "jitter 1e-6",
        "tol 1e-6",
        f"log_marginal {out['log_marginal']:.16e}",
        f"converged {int(out['converged'])}",
        f"iters {out['iters']}",
        "X " + " ".join(f"{x[0]:.16e}" for x in X),
        "Y " + " ".join(f"{y:.1f}" for y in Y),
        "f_hat " + " ".join(f"{v:.16e}" for v in out["f_hat"]),
    ]
    fixture.write_text("\n".join(lines) + "\n")
    print(f"Wrote {fixture}")
    print(f"log_marginal={out['log_marginal']:.16e} converged={out['converged']} iters={out['iters']}")


if __name__ == "__main__":
    main()
