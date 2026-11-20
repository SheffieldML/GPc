#!/usr/bin/env python3
"""Generate text fixtures for CIP-0018 MRD vs GPy.

Requires GPy (maintainer tooling; committed fixtures mean CI does not).

CMrd is built on CBayesGplvm (CIP-0017 VarDTC). GPy.models.MRD currently
assembles BayesianGPLVMMiniBatch children whose VarDTC term drifts slightly
from BayesianGPLVM; the accepted ELBO here is therefore the single-KL
assembly of two BayesianGPLVM views that share (mu, S, Z):

    ELBO = L0 + L1 + KL(q||p)
         = sum_v L_VarDTC^(v) - KL

which is what CMrd computes and what GPy MRD intends via kl_factr=1/V.

Example:
  MPLCONFIGDIR=/tmp/mpl /tmp/gpc-gpy-fix/bin/python scripts/gen_cip0018_mrd_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "cip0018"
SEED = 17
N, Q, M = 8, 2, 3
D0, D1 = 2, 3
BETA0, BETA1 = 2.0, 2.5
VARIANCE0, VARIANCE1 = 1.0, 1.2
LENGTHSCALE0, LENGTHSCALE1 = 1.0, 0.8  # GPy; GPc inverseWidth = 1/ls**2


def write_matrix(f, name: str, A: np.ndarray):
    A = np.asarray(A, dtype=float)
    f.write(f"{name} {A.shape[0]} {A.shape[1] if A.ndim > 1 else 1}\n")
    flat = A.reshape(-1, order="C")
    f.write(" ".join(f"{v:.16e}" for v in flat) + "\n")


def make_views(rng: np.random.Generator):
    Xtrue = rng.normal(size=(N, Q))
    Y0 = np.column_stack(
        [
            Xtrue[:, 0],
            0.3 * Xtrue[:, 1] + 0.05 * rng.normal(size=N),
        ]
    )
    Y1 = np.column_stack(
        [
            Xtrue[:, 0],
            0.3 * Xtrue[:, 1] + 0.05 * rng.normal(size=N),
            0.1 * Xtrue[:, 0] + 0.05 * rng.normal(size=N),
        ]
    )
    return Y0, Y1


def main():
    try:
        import GPy
        from GPy.models import BayesianGPLVM
        from GPy.kern import RBF
        from GPy.likelihoods import Gaussian
        from GPy.core.parameterization.variational import NormalPrior
    except ImportError as exc:
        raise SystemExit(
            "GPy is required to regenerate fixtures (not needed at CI runtime)."
        ) from exc

    rng = np.random.default_rng(SEED)
    Y0, Y1 = make_views(rng)
    mu = rng.normal(size=(N, Q))
    S = np.full((N, Q), 0.3)
    Z = np.zeros((M, Q))
    for m in range(M):
        n = (m * N) // M
        Z[m] = mu[n]

    k0 = RBF(Q, variance=VARIANCE0, lengthscale=LENGTHSCALE0, ARD=False)
    k1 = RBF(Q, variance=VARIANCE1, lengthscale=LENGTHSCALE1, ARD=False)
    b0 = BayesianGPLVM(
        Y0,
        Q,
        X=mu.copy(),
        X_variance=S.copy(),
        Z=Z.copy(),
        kernel=k0,
        likelihood=Gaussian(variance=1.0 / BETA0),
        num_inducing=M,
    )
    b1 = BayesianGPLVM(
        Y1,
        Q,
        X=mu.copy(),
        X_variance=S.copy(),
        Z=Z.copy(),
        kernel=k1,
        likelihood=Gaussian(variance=1.0 / BETA1),
        num_inducing=M,
    )
    b0.parameters_changed()
    b1.parameters_changed()

    L0 = float(np.asarray(b0.log_likelihood()).reshape(-1)[0])
    L1 = float(np.asarray(b1.log_likelihood()).reshape(-1)[0])
    KL = float(NormalPrior().KL_divergence(b0.X))
    # Each BayesianGPLVM already subtracts a full KL; undo one to match MRD.
    elbo = L0 + L1 + KL

    iw0 = 1.0 / (LENGTHSCALE0**2)
    iw1 = 1.0 / (LENGTHSCALE1**2)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mrd_rbf_tiny.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("# CIP-0018 fixture: GPy BayesianGPLVM pair, single-KL MRD ELBO\n")
        f.write(
            f"# source GPy {GPy.__version__} "
            f"VarDTC.const_jitter={b0.inference_method.const_jitter}\n"
        )
        f.write(
            "# ELBO = L0 + L1 + KL (undo one of two full KLs); matches CMrd / "
            "GPy MRD intent (kl_factr=1/V). Not raw GPy.models.MRD MiniBatch.\n"
        )
        f.write(f"# seed {SEED}\n")
        f.write(f"# L0 {L0:.16e} L1 {L1:.16e} KL {KL:.16e}\n")
        f.write(f"N {N}\nD0 {D0}\nD1 {D1}\nQ {Q}\nM {M}\n")
        f.write(f"beta0 {BETA0:.16e}\n")
        f.write(f"beta1 {BETA1:.16e}\n")
        f.write(f"inverseWidth0 {iw0:.16e}\n")
        f.write(f"variance0 {VARIANCE0:.16e}\n")
        f.write(f"inverseWidth1 {iw1:.16e}\n")
        f.write(f"variance1 {VARIANCE1:.16e}\n")
        write_matrix(f, "Y0", Y0)
        write_matrix(f, "Y1", Y1)
        write_matrix(f, "mu", mu)
        write_matrix(f, "S", S)
        write_matrix(f, "Z", Z)
        f.write(f"elbo {elbo:.16e}\n")
    print(f"Wrote {path} elbo={elbo} (L0={L0} L1={L1} KL={KL})")


if __name__ == "__main__":
    main()
