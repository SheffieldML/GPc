#!/usr/bin/env python3
"""Generate text fixtures for CIP-0017 Phase 3 Bayesian GP-LVM vs GPy.

Requires GPy (maintainer tooling; committed fixtures mean CI does not).
Example:
  MPLCONFIGDIR=/tmp/mpl \\
    /path/to/GPy/.venv/bin/python scripts/gen_cip0017_bayes_gplvm_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "cip0017"
SEED = 17
N, D, Q, M = 6, 3, 2, 3
BETA = 2.0
VARIANCE = 1.0
LENGTHSCALE = 1.0  # GPy; GPc inverseWidth = 1/lengthscale**2


def make_Y():
    # Match testBayesGplvm.cpp makeY.
    Y = np.zeros((N, D))
    for n in range(N):
        for d in range(D):
            Y[n, d] = np.sin(0.3 * n + 0.7 * d) + 0.1 * n
    return Y


def write_matrix(f, name: str, A: np.ndarray):
    A = np.asarray(A, dtype=float)
    f.write(f"{name} {A.shape[0]} {A.shape[1] if A.ndim > 1 else 1}\n")
    flat = A.reshape(-1, order="C")
    f.write(" ".join(f"{v:.16e}" for v in flat) + "\n")


def write_vector(f, name: str, v: np.ndarray):
    v = np.asarray(v, dtype=float).reshape(-1, order="C")
    f.write(f"{name} {v.size}\n")
    f.write(" ".join(f"{x:.16e}" for x in v) + "\n")


def main():
    try:
        import GPy
        from GPy.models import BayesianGPLVM
        from GPy.kern import RBF
        from GPy.likelihoods import Gaussian
    except ImportError as exc:
        raise SystemExit(
            "GPy is required to regenerate fixtures (not needed at CI runtime)."
        ) from exc

    rng = np.random.default_rng(SEED)
    Y = make_Y()
    mu = rng.normal(size=(N, Q))
    S = np.full((N, Q), 0.3)
    # Evenly spaced subset of mu rows (same spirit as CBayesGplvm::initZFromMu).
    Z = np.zeros((M, Q))
    for m in range(M):
        n = (m * N) // M
        Z[m] = mu[n]

    iw = 1.0 / (LENGTHSCALE ** 2)
    kern = RBF(Q, variance=VARIANCE, lengthscale=LENGTHSCALE, ARD=False)
    lik = Gaussian(variance=1.0 / BETA)
    model = BayesianGPLVM(
        Y,
        Q,
        X=mu.copy(),
        X_variance=S.copy(),
        Z=Z.copy(),
        kernel=kern,
        likelihood=lik,
        num_inducing=M,
    )
    # Ensure gradients / psi side-effects are current.
    model.parameters_changed()

    psi0 = np.asarray(model.kern.psi0(model.Z, model.X), dtype=float).reshape(-1)
    psi1 = np.asarray(model.kern.psi1(model.Z, model.X), dtype=float)
    psi2 = np.asarray(model.kern.psi2(model.Z, model.X), dtype=float)
    elbo = float(np.asarray(model.log_likelihood()).reshape(-1)[0])

    # Natural-space grads in GPc CBayesGplvm packing:
    #   inverseWidth, variance, beta, Z(row-major), mu, S
    g_var = float(model.kern.variance.gradient.reshape(-1)[0])
    g_ls = float(model.kern.lengthscale.gradient.reshape(-1)[0])
    # γ = 1/l^2 ⇒ ∂L/∂γ = ∂L/∂l / (dγ/dl) = g_ls / (-2 l^{-3}) = -0.5 g_ls l^3
    g_iw = -0.5 * g_ls * (LENGTHSCALE ** 3)
    g_noise = float(model.likelihood.variance.gradient.reshape(-1)[0])
    # β = 1/σ² ⇒ ∂L/∂β = ∂L/∂σ² * (-1/β²)
    g_beta = g_noise * (-1.0 / (BETA ** 2))
    g_Z = np.asarray(model.Z.gradient, dtype=float)
    g_mu = np.asarray(model.X.mean.gradient, dtype=float)
    g_S = np.asarray(model.X.variance.gradient, dtype=float)
    g_nat = np.concatenate(
        [
            np.array([g_iw, g_var, g_beta]),
            g_Z.reshape(-1, order="C"),
            g_mu.reshape(-1, order="C"),
            g_S.reshape(-1, order="C"),
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bayes_gplvm_rbf_tiny.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("# CIP-0017 Phase 3 fixture: GPy BayesianGPLVM vs CBayesGplvm\n")
        f.write(f"# source GPy {GPy.__version__} VarDTC.const_jitter="
                f"{model.inference_method.const_jitter}\n")
        f.write(f"# seed {SEED}\n")
        f.write(f"N {N}\nD {D}\nQ {Q}\nM {M}\n")
        f.write(f"beta {BETA:.16e}\n")
        f.write(f"inverseWidth {iw:.16e}\n")
        f.write(f"variance {VARIANCE:.16e}\n")
        write_matrix(f, "Y", Y)
        write_matrix(f, "mu", mu)
        write_matrix(f, "S", S)
        write_matrix(f, "Z", Z)
        write_vector(f, "psi0", psi0)
        write_matrix(f, "psi1", psi1)
        write_matrix(f, "psi2", psi2)
        f.write(f"elbo {elbo:.16e}\n")
        write_vector(f, "g_natural", g_nat)

    print(f"wrote {path}")
    print(f"elbo={elbo:.12e} psi0_sum={psi0.sum():.12e}")


if __name__ == "__main__":
    main()
