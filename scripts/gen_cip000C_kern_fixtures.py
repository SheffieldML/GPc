#!/usr/bin/env python3
"""Generate matio fixtures for CIP-000C kernels (numpy reference + FD grads).

Formulas match GPc C++ / GPy StdPeriodic, Cosine, Brownian where noted.
Params in the .mat file are in *transformed* space (exp / sigmoid), matching
testKern's setTransParams / getGradTransParams contract.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "matfiles"
EPS = 1e-16
LIM = 36.0
FD = 1e-6
RNG = np.random.default_rng(20261115)


def atox_exp(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.exp(np.clip(a, -LIM, LIM))


def xtoa_exp(x: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(x, dtype=float))


def atox_sig(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = 1.0 / (1.0 + np.exp(-np.clip(a, -LIM, LIM)))
    return np.clip(out, EPS, 1.0 - EPS)


def xtoa_sig(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def gradfact_exp(x: float) -> float:
    return float(x)


def gradfact_sig(x: float) -> float:
    return float(x * (1.0 - x))


def ard_r2(X: np.ndarray, X2: np.ndarray, scales: np.ndarray) -> np.ndarray:
    # (n, m) ARD squared distances
    d = X[:, None, :] - X2[None, :, :]
    return np.sum((d * d) * scales[None, None, :], axis=-1)


def make_xy(n: int, m: int, d: int, positive: bool = False):
    X = RNG.normal(size=(n, d))
    X2 = RNG.normal(size=(m, d))
    if positive:
        X = np.abs(X) + 0.1
        X2 = np.abs(X2) + 0.1
    return X, X2


def fd_param_grads(K_fn, params, transforms, X, X2, covGrad, h=FD):
    """Finite-difference grads in transformed space (matches getGradTransParams)."""
    g = np.zeros(len(params))
    for i in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        Kp = K_fn(p_plus, X, X2)
        Km = K_fn(p_minus, X, X2)
        g[i] = np.sum(covGrad * (Kp - Km) / (2 * h))
    return g


def fd_grad_X(K_elem, params, X, X2, h=1e-4):
    """G2 cell: for each row i of X, matrix (m, d) of dK(X_i, X2_k)/dX_i.

    Confirmed against the existing, tested CRbfardKern::getGradX (val +=
    pf*(X2-X)*exp*scales): with K(x,x2)=exp(-(x-x2)^2), dK/dx at x=0,x2=1 is
    +0.7358 while dK/dx2 is -0.7358. The C++ convention differentiates with
    respect to X (the `row` argument), so we perturb X here, not X2.

    Uses a 5-point (4th order) central stencil rather than the 2-point
    stencil used elsewhere: testKern.cpp compares G2 against the analytic
    getGradX at the strict ndlutil::MATCHTOL (1e-10, no looser override), so
    the O(h^2) truncation/round-off of a 2-point stencil at h=FD=1e-6 was
    landing right at ~1e-10 and intermittently failing. The O(h^4) stencil
    at h=1e-4 pushes truncation error far below MATCHTOL while keeping
    round-off (~eps/h) small too.
    """
    n, d = X.shape
    m = X2.shape[0]
    cells = []
    for i in range(n):
        G = np.zeros((m, d))
        for k in range(m):
            for j in range(d):
                Xpp = X.copy(); Xp = X.copy(); Xm = X.copy(); Xmm = X.copy()
                Xpp[i, j] += 2 * h
                Xp[i, j] += h
                Xm[i, j] -= h
                Xmm[i, j] -= 2 * h
                kpp = K_elem(params, Xpp[i : i + 1], X2[k : k + 1])[0, 0]
                kp = K_elem(params, Xp[i : i + 1], X2[k : k + 1])[0, 0]
                km = K_elem(params, Xm[i : i + 1], X2[k : k + 1])[0, 0]
                kmm = K_elem(params, Xmm[i : i + 1], X2[k : k + 1])[0, 0]
                G[k, j] = (-kpp + 8 * kp - 8 * km + kmm) / (12 * h)
        cells.append(G)
    return cells


def pack_fixture(name: str, params_t, transforms, K_fn, X, X2, stationary_diag_grad=True):
    """params_t: transformed params; transforms: list of 'exp'|'sig' per param."""
    params_t = np.asarray(params_t, dtype=float).ravel()
    n = X.shape[0]
    m = X2.shape[0]

    def K_from_t(pt, A, B):
        return K_fn(pt, A, B)

    K2 = K_from_t(params_t, X, X)
    K4 = K_from_t(params_t, X, X2)
    k2 = np.diag(K2)[:, None]
    covGrad = RNG.normal(size=(n, n))
    covGrad = 0.5 * (covGrad + covGrad.T)
    covGrad2 = RNG.normal(size=(n, m))

    g2 = fd_param_grads(K_from_t, params_t, transforms, X, X, covGrad)
    g4 = fd_param_grads(K_from_t, params_t, transforms, X, X2, covGrad2)
    G2 = fd_grad_X(K_from_t, params_t, X, X2)
    if stationary_diag_grad:
        GD2 = np.zeros_like(X)
    else:
        # Brownian diag depends on |x|
        GD2 = np.zeros_like(X)
        for i in range(n):
            for j in range(X.shape[1]):
                Xp = X.copy()
                Xm = X.copy()
                Xp[i, j] += FD
                Xm[i, j] -= FD
                # diag K = variance * |x|
                # filled by kernel-specific override below if needed
                GD2[i, j] = 0.0

    # MATLAB cell: 1 x n object array
    G2_cell = np.empty((1, n), dtype=object)
    for i in range(n):
        G2_cell[0, i] = np.ascontiguousarray(G2[i], dtype=np.float64)

    savemat(
        OUT / f"{name}KernTest.mat",
        {
            "X": np.ascontiguousarray(X, dtype=np.float64),
            "X2": np.ascontiguousarray(X2, dtype=np.float64),
            "params": params_t.reshape(1, -1),
            "K2": np.ascontiguousarray(K2, dtype=np.float64),
            "K4": np.ascontiguousarray(K4, dtype=np.float64),
            "k2": np.ascontiguousarray(k2, dtype=np.float64),
            "covGrad": np.ascontiguousarray(covGrad, dtype=np.float64),
            "covGrad2": np.ascontiguousarray(covGrad2, dtype=np.float64),
            "g2": g2.reshape(1, -1),
            "g4": g4.reshape(1, -1),
            "G2": G2_cell,
            "GD2": np.ascontiguousarray(GD2, dtype=np.float64),
        },
        do_compression=False,
    )
    print(f"wrote {name}KernTest.mat  params={params_t.size} N={n} M={m} D={X.shape[1]}")


# ---- Kernel K(params_transformed, X, X2) ----


def decode_ard_ls(pt, d):
    # lengthScale, variance, scales[d]
    ls = atox_exp(pt[0])
    var = atox_exp(pt[1])
    scales = atox_sig(pt[2 : 2 + d])
    return ls, var, scales


def K_matern32ard(pt, X, X2):
    d = X.shape[1]
    ls, var, scales = decode_ard_ls(pt, d)
    r2 = ard_r2(X, X2, scales)
    wi2 = 3.0 / (ls * ls)
    t = np.sqrt(np.maximum(r2 * wi2, 0.0))
    return var * (1.0 + t) * np.exp(-t)


def K_matern52ard(pt, X, X2):
    d = X.shape[1]
    ls, var, scales = decode_ard_ls(pt, d)
    r2 = ard_r2(X, X2, scales)
    wi2 = 5.0 / (ls * ls)
    n2wi2 = r2 * wi2
    t = np.sqrt(np.maximum(n2wi2, 0.0))
    return var * (1.0 + t + n2wi2 / 3.0) * np.exp(-t)


def K_ratquadard(pt, X, X2):
    d = X.shape[1]
    alpha = atox_exp(pt[0])
    ls = atox_exp(pt[1])
    var = atox_exp(pt[2])
    scales = atox_sig(pt[3 : 3 + d])
    r2 = ard_r2(X, X2, scales)
    wi2 = 0.5 / (ls * ls * alpha)
    return var * np.power(1.0 + r2 * wi2, -alpha)


def K_exp(pt, X, X2):
    iw = atox_exp(pt[0])
    var = atox_exp(pt[1])
    r2 = np.sum((X[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return var * np.exp(-np.sqrt(np.maximum(r2, 0.0)) * iw)


def K_stdperiodic(pt, X, X2):
    # period, lengthScale, variance — isotropic (CIP-000C v1)
    period = atox_exp(pt[0])
    ls = atox_exp(pt[1])
    var = atox_exp(pt[2])
    base = np.pi * (X[:, None, :] - X2[None, :, :]) / period
    return var * np.exp(-0.5 * np.sum((np.sin(base) / ls) ** 2, axis=-1))


def K_cosine(pt, X, X2):
    # lengthScale, variance — GPy Cosine: σ² cos(r), r = sqrt(sum ((dx)/ℓ)²)
    ls = atox_exp(pt[0])
    var = atox_exp(pt[1])
    r2 = np.sum(((X[:, None, :] - X2[None, :, :]) / ls) ** 2, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    return var * np.cos(r)


def K_sinc(pt, X, X2):
    # CIP-0038: GPy Sinc — σ² * np.sinc(2 r)
    ls = atox_exp(pt[0])
    var = atox_exp(pt[1])
    r2 = np.sum(((X[:, None, :] - X2[None, :, :]) / ls) ** 2, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    return var * np.sinc(2.0 * r)


def K_expquadcosine(pt, X, X2):
    # CIP-0038: lengthScale, variance, period — σ² exp(-2π² r²) cos(2π r / T)
    ls = atox_exp(pt[0])
    var = atox_exp(pt[1])
    period = atox_exp(pt[2])
    r2 = np.sum(((X[:, None, :] - X2[None, :, :]) / ls) ** 2, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    return var * np.exp(-2.0 * np.pi**2 * r**2) * np.cos(2.0 * np.pi * r / period)


def K_brownian(pt, X, X2):
    # variance only; 1-D GPy Brownian
    var = atox_exp(pt[0])
    x = X[:, 0:1]
    y = X2[:, 0:1].T
    same = np.sign(x) * np.sign(y) > 0
    # also treat zeros carefully: sign(0)=0 → product 0 → False → 0 cov
    return var * np.where(same, np.minimum(np.abs(x), np.abs(y)), 0.0)


def brownian_diag_grad(pt, X):
    var = atox_exp(pt[0])
    g = np.zeros_like(X)
    # d/dx (var*|x|) = var * sign(x) for x!=0
    g[:, 0] = var * np.sign(X[:, 0])
    g[X[:, 0] == 0, 0] = 0.0
    return g


def main():
    OUT.mkdir(exist_ok=True)
    n, m, d = 12, 16, 2

    # matern32ard
    X, X2 = make_xy(n, m, d)
    pt = np.array([-0.2, 0.1, -0.5, 0.3])  # ls, var, s0, s1
    pack_fixture("matern32ard", pt, ["exp", "exp", "sig", "sig"], K_matern32ard, X, X2)

    # matern52ard
    X, X2 = make_xy(n, m, d)
    pt = np.array([0.05, -0.15, -0.2, 0.4])
    pack_fixture("matern52ard", pt, ["exp", "exp", "sig", "sig"], K_matern52ard, X, X2)

    # ratquadard
    X, X2 = make_xy(n, m, d)
    pt = np.array([0.0, -0.1, 0.2, -0.3, 0.25])  # alpha, ls, var, s0, s1
    pack_fixture("ratquadard", pt, ["exp", "exp", "exp", "sig", "sig"], K_ratquadard, X, X2)

    # exp
    X, X2 = make_xy(n, m, d)
    pt = np.array([-0.3, 0.05])
    pack_fixture("exp", pt, ["exp", "exp"], K_exp, X, X2)

    # stdperiodic
    X, X2 = make_xy(n, m, d)
    pt = np.array([0.2, -0.1, 0.0])  # period, ls, var
    pack_fixture("stdperiodic", pt, ["exp", "exp", "exp"], K_stdperiodic, X, X2)

    # cosine
    X, X2 = make_xy(n, m, d)
    pt = np.array([-0.05, 0.1])
    pack_fixture("cosine", pt, ["exp", "exp"], K_cosine, X, X2)

    # sinc (CIP-0038)
    X, X2 = make_xy(n, m, d)
    pt = np.array([-0.08, 0.12])
    pack_fixture("sinc", pt, ["exp", "exp"], K_sinc, X, X2)

    # expquadcosine (CIP-0038)
    X, X2 = make_xy(n, m, d)
    pt = np.array([-0.05, 0.1, 0.25])  # ls, var, period
    pack_fixture("expquadcosine", pt, ["exp", "exp", "exp"], K_expquadcosine, X, X2)

    # brownian (1-D, positive-ish)
    X, X2 = make_xy(n, m, 1, positive=True)
    # mix signs for GPy same-sign rule
    X[::3, 0] *= -1
    X2[1::4, 0] *= -1
    pt = np.array([0.15])

    # custom pack for brownian diag grad
    params_t = pt
    K2 = K_brownian(params_t, X, X)
    K4 = K_brownian(params_t, X, X2)
    k2 = np.diag(K2)[:, None]
    covGrad = 0.5 * (RNG.normal(size=(n, n)) + RNG.normal(size=(n, n)).T)
    covGrad2 = RNG.normal(size=(n, m))
    g2 = fd_param_grads(K_brownian, params_t, ["exp"], X, X, covGrad)
    g4 = fd_param_grads(K_brownian, params_t, ["exp"], X, X2, covGrad2)
    G2 = fd_grad_X(K_brownian, params_t, X, X2)
    GD2 = brownian_diag_grad(params_t, X)
    G2_cell = np.empty((1, n), dtype=object)
    for i in range(n):
        G2_cell[0, i] = np.ascontiguousarray(G2[i], dtype=np.float64)
    savemat(
        OUT / "brownianKernTest.mat",
        {
            "X": X,
            "X2": X2,
            "params": params_t.reshape(1, -1),
            "K2": K2,
            "K4": K4,
            "k2": k2,
            "covGrad": covGrad,
            "covGrad2": covGrad2,
            "g2": g2.reshape(1, -1),
            "g4": g4.reshape(1, -1),
            "G2": G2_cell,
            "GD2": GD2,
        },
    )
    print(f"wrote brownianKernTest.mat  N={n} M={m} D=1")


if __name__ == "__main__":
    main()
