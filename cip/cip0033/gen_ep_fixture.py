#!/usr/bin/env python3
"""CIP-0033: numpy twin of CEpInference (GPy EP formulas) for frozen fixtures.

No SciPy dependency — uses math.erf. Regenerates cip/cip0033/ep_ftc_fixture.txt.

Usage:
  python3 cip/cip0033/gen_ep_fixture.py
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


def bernoulli_probit_ep_moments(y: float, tau: float, v: float):
    sign = 1.0 if y > 0.5 else -1.0
    den = math.sqrt(tau * tau + tau)
    z = sign * v / den
    r = grad_ln_cum_gaussian(z)
    Zhat = math.exp(ln_cum_gaussian(z))
    mu = v / tau + sign * r / den
    s2 = 1.0 / tau - (r / (tau * tau + tau)) * (z + r)
    return Zhat, mu, s2


def rbf_gram(X: np.ndarray, variance: float, lengthscale: float, jitter: float) -> np.ndarray:
    d2 = (X[:, None, :] - X[None, :, :]) ** 2
    r2 = np.sum(d2, axis=-1) / (lengthscale ** 2)
    K = variance * np.exp(-0.5 * r2)
    return K + jitter * np.eye(K.shape[0])


def ep_run(K: np.ndarray, Y: np.ndarray, max_iters=100, eta=1.0, delta=1.0, tol=1e-8):
    n = K.shape[0]
    tau = np.zeros(n)
    v = np.zeros(n)
    m0 = np.zeros(n)
    eps = 1e-12

    def recompute():
        sqrt_tau = np.sqrt(np.maximum(tau, 0.0))
        B = np.eye(n) + (sqrt_tau[:, None] * K) * sqrt_tau[None, :]
        U = np.linalg.cholesky(B).T
        logDetB = 2.0 * np.sum(np.log(np.diag(U)))
        V = np.linalg.solve(U.T, (sqrt_tau[:, None] * K))
        Sigma = K - V.T @ V
        var_diag = np.clip(np.diag(Sigma), eps, None)
        rhs = sqrt_tau * (K @ v + m0)
        aux = np.linalg.solve(U, np.linalg.solve(U.T, rhs))
        alpha = v - sqrt_tau * aux
        mu = K @ alpha + m0
        return Sigma, var_diag, mu, logDetB

    Sigma, var_diag, mu, logDetB = recompute()
    Zhat = np.zeros(n)
    tau_cav = np.zeros(n)
    v_cav = np.zeros(n)
    converged = False
    iters = 0
    for it in range(max_iters):
        tau_old, v_old = tau.copy(), v.copy()
        for i in range(n):
            sii, mui = var_diag[i], mu[i]
            t_cav = 1.0 / sii - eta * tau[i]
            v_c = mui / sii - eta * v[i]
            if t_cav <= eps:
                t_cav = eps
            tau_cav[i], v_cav[i] = t_cav, v_c
            Zh, mu_hat, s2_hat = bernoulli_probit_ep_moments(Y[i], t_cav, v_c)
            Zhat[i] = Zh
            d_tau = (delta / eta) * (1.0 / s2_hat - 1.0 / sii)
            d_v = (delta / eta) * (mu_hat / s2_hat - mui / sii)
            tau[i] = max(tau[i] + d_tau, eps)
            v[i] = v[i] + d_v
        Sigma, var_diag, mu, logDetB = recompute()
        iters = it + 1
        if it > 0 and max(np.max(np.abs(tau - tau_old)), np.max(np.abs(v - v_old))) < tol:
            converged = True
            break

    log_Z_tilde = 0.0
    for i in range(n):
        tS, vS, tC, vC, Zh = tau[i], v[i], tau_cav[i], v_cav[i], max(Zhat[i], eps)
        den = tC + tS
        log_Z_tilde += (
            math.log(Zh)
            + 0.5 * math.log(2.0 * math.pi)
            + 0.5 * math.log(1.0 + tS / tC)
            - 0.5 * (vS * vS) / den
            + 0.5 * (vC * ((tS / tC) * vC - 2.0 * vS) / den)
        )
    v_centered = v - tau * m0
    SigmaV = Sigma @ v_centered
    quad = float(v_centered @ SigmaV)
    mean_term = float(m0 @ (tau * m0 - 2.0 * v))
    log_marginal = 0.5 * (-n * math.log(2.0 * math.pi) - logDetB + quad - mean_term) + log_Z_tilde
    return {
        "log_marginal": log_marginal,
        "converged": converged,
        "iters": iters,
        "tau": tau,
        "v": v,
        "mu": mu,
    }


def main():
    X = np.array([[-1.5], [-0.5], [0.5], [1.5]], dtype=float)
    Y = np.array([1.0, 1.0, 0.0, 0.0])
    K = rbf_gram(X, variance=1.0, lengthscale=1.0, jitter=1e-6)
    out = ep_run(K, Y, tol=1e-8)
    root = Path(__file__).resolve().parent
    fixture = root / "ep_ftc_fixture.txt"
    lines = [
        "# CIP-0033 FTC EP fixture (numpy twin of CEpInference)",
        "N 4",
        "variance 1.0",
        "lengthscale 1.0",
        "jitter 1e-6",
        "tol 1e-8",
        f"log_marginal {out['log_marginal']:.16e}",
        f"converged {int(out['converged'])}",
        f"iters {out['iters']}",
        "X " + " ".join(f"{x[0]:.16e}" for x in X),
        "Y " + " ".join(f"{y:.1f}" for y in Y),
        "tau " + " ".join(f"{t:.16e}" for t in out["tau"]),
        "v " + " ".join(f"{vi:.16e}" for vi in out["v"]),
    ]
    fixture.write_text("\n".join(lines) + "\n")
    print(f"Wrote {fixture}")
    print(f"log_marginal={out['log_marginal']:.16e} converged={out['converged']} iters={out['iters']}")


if __name__ == "__main__":
    main()
