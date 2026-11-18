#!/usr/bin/env python3
"""CIP-001B offline prototype: LQ bound vs Gram+chol vs dense VFE on small N.

Real-world stand-in: a small, well-conditioned sparse GP / VFE problem (modest
N, random whitened features A). Checks that LQ of [I,A] (and y-augmented r^2)
reproduces Gram+chol before trusting the C++ path on harder data.

Reference: Seeger, Hetzel, Dai, Meissner & Lawrence (2017),
Auto-differentiating linear algebra, arXiv:1710.08717 §3.3 (BLR via LQ of
[I, α^{1/2} X]) applied to the collapsed VFE/DTC bound.
"""
from __future__ import annotations

import numpy as np


def lq_factor(W: np.ndarray):
    """Return L (lower, positive diag) and Q with W = L @ Q (rows of Q orthonormal)."""
    # NumPy has no gelqf; use QR of W.T: W.T = Qr Rr ⇒ W = Rr.T Qr.T
    Qr, Rr = np.linalg.qr(W.T, mode="reduced")  # W.T = Qr @ Rr, Rr upper (n×m wait)
    # W is m×n, W.T is n×m. qr reduced: Qr n×m, Rr m×m upper.
    L = Rr.T  # m×m lower
    Q = Qr.T  # m×n
    # Sign-normalise
    s = np.sign(np.diag(L))
    s[s == 0] = 1.0
    L = L * s[:, None]
    Q = s[:, None] * Q
    return L, Q


def bound_pieces(A: np.ndarray, y: np.ndarray, sigma: float, trKnn: float):
    """Return (logdet_chol, logdet_lq, r2_woodbury, r2_lq, froA)."""
    m, n = A.shape
    C = np.eye(m) + A @ A.T
    Lc = np.linalg.cholesky(C)
    logdet_chol = 2.0 * np.sum(np.log(np.abs(np.diag(Lc))))

    W = np.hstack([np.eye(m), A])
    L, Q = lq_factor(W)
    logdet_lq = 2.0 * np.sum(np.log(np.abs(np.diag(L))))

    ys = y / sigma
    # Woodbury quadratic: β||y||^2 - ||L^{-1} A y||^2 with A already σ-scaled features
    c = np.linalg.solve(L, A @ ys)
    r2_woodbury = float(ys @ ys - c @ c)

    Wa = np.vstack([W, np.concatenate([np.zeros(m), ys])])
    La, _Qa = lq_factor(Wa)
    r2_lq = float(La[-1, -1] ** 2)

    froA = float(np.linalg.norm(A, "fro") ** 2)
    # Also ||L||_F^2 - M identity
    fro_from_L = float(np.linalg.norm(L, "fro") ** 2 - m)
    return logdet_chol, logdet_lq, r2_woodbury, r2_lq, froA, fro_from_L


def main() -> None:
    rng = np.random.default_rng(0)
    m, n = 5, 40
    sigma = 0.3
    A = rng.normal(size=(m, n)) / np.sqrt(n)
    y = rng.normal(size=n)
    trKnn = float(np.sum(rng.uniform(0.5, 1.5, size=n)))
    ld_c, ld_l, r2_w, r2_l, froA, froL = bound_pieces(A, y, sigma, trKnn)
    print(f"logdet chol={ld_c:.16f}")
    print(f"logdet lq  ={ld_l:.16f}  |diff|={abs(ld_c-ld_l):.3e}")
    print(f"r2 woodbury={r2_w:.16f}")
    print(f"r2 lq      ={r2_l:.16f}  |diff|={abs(r2_w-r2_l):.3e}")
    print(f"||A||_F^2  ={froA:.16f}")
    print(f"||L||_F^2-M={froL:.16f}  |diff|={abs(froA-froL):.3e}")
    assert abs(ld_c - ld_l) < 1e-12
    assert abs(r2_w - r2_l) < 1e-10
    assert abs(froA - froL) < 1e-10
    print("prototype OK")


if __name__ == "__main__":
    main()
