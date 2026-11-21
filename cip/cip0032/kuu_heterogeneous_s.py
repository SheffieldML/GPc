"""CIP-0032 Phase A: heterogeneous-s ridge K_uu oracle.

Tied (s_m ≡ s) must match CIP-0028. General pair formula uses
  alpha_m = 1/s_m^2 - 1/2,  alpha_n = 1/s_n^2 - 1/2,
  det(I + A G) with A = diag(alpha_m, alpha_n), G = [[1,c],[c,1]].
"""
from __future__ import annotations

import numpy as np


def kuu_pair(v_m, a_m, s_m, v_n, a_n, s_n, Dp: int) -> float:
    v_m = np.asarray(v_m, float)
    v_n = np.asarray(v_n, float)
    v_m = v_m / np.linalg.norm(v_m)
    v_n = v_n / np.linalg.norm(v_n)
    c = float(np.clip(v_m @ v_n, -1.0, 1.0))
    s2m, s2n = s_m * s_m, s_n * s_n
    am, an = float(a_m), float(a_n)
    alpha_m = 1.0 / s2m - 0.5
    alpha_n = 1.0 / s2n - 0.5
    det_i = (1.0 + alpha_m) * (1.0 + alpha_n) - alpha_m * alpha_n * c * c
    minv11 = (1.0 + alpha_n) / det_i
    minv12 = -alpha_m * c / det_i
    minv21 = -alpha_n * c / det_i
    minv22 = (1.0 + alpha_m) / det_i
    h11 = minv11 + c * minv21
    h12 = minv12 + c * minv22
    h21 = c * minv11 + minv21
    h22 = c * minv12 + minv22
    bm, bn = am / s2m, an / s2n
    quad = h11 * bm * bm + h22 * bn * bn + (h12 + h21) * bm * bn
    lognorm = (
        -0.25 * np.log(np.pi * s2m)
        -0.25 * np.log(np.pi * s2n)
        -0.5 * (Dp - 1) * np.log(2.0 * np.pi)
    )
    const = -am * am / (2.0 * s2m) - an * an / (2.0 * s2n)
    log_k = (
        lognorm
        + 0.5 * Dp * np.log(2.0 * np.pi)
        - 0.5 * np.log(det_i)
        + 0.5 * quad
        + const
    )
    return float(np.exp(log_k))


def kuu_tied_cip0028(V, a, s: float) -> np.ndarray:
    """Reference tied formula (prototype ridge.py)."""
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    M, Dp = V.shape
    s2 = s * s
    alpha = 1.0 / s2 - 0.5
    G12 = V @ V.T
    lognorm2 = -0.5 * np.log(np.pi * s2) - (Dp - 1) / 2.0 * np.log(2.0 * np.pi)
    am, an = a[:, None], a[None, :]
    c = G12
    det_i = (1 + alpha) ** 2 - (alpha * c) ** 2
    inv11 = (1 + alpha) / det_i
    inv12 = -alpha * c / det_i
    h11 = inv11 + c * inv12
    h12 = inv12 + c * inv11
    bm, bn = am / s2, an / s2
    quad = h11 * (bm**2 + bn**2) + 2.0 * h12 * bm * bn
    const = -(am**2 + an**2) / (2.0 * s2)
    log_k = lognorm2 + Dp / 2.0 * np.log(2.0 * np.pi) - 0.5 * np.log(det_i) + 0.5 * quad + const
    K = np.exp(log_k)
    return 0.5 * (K + K.T)


def kuu_heterogeneous(V, a, s_vec) -> np.ndarray:
    V = np.asarray(V, float)
    a = np.asarray(a, float).reshape(-1)
    s_vec = np.asarray(s_vec, float).reshape(-1)
    M, Dp = V.shape
    K = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            kij = kuu_pair(V[i], a[i], s_vec[i], V[j], a[j], s_vec[j], Dp)
            K[i, j] = kij
            K[j, i] = kij
    return K


def _quad_1d(v1, a1, s1, v2, a2, s2, n=1401, span=10.0) -> float:
    """2-D quadrature of <psi_1, psi_2> for D'=2."""
    g = np.linspace(-span, span, n)
    W0, W1 = np.meshgrid(g, g, indexing="ij")

    def psi(v, a, s):
        v = v / np.linalg.norm(v)
        t = W0 * v[0] + W1 * v[1]
        perp = -W0 * v[1] + W1 * v[0]
        return (
            (np.pi * s * s) ** -0.25
            * np.exp(-((t - a) ** 2) / (2.0 * s * s))
            * (2.0 * np.pi) ** -0.25
            * np.exp(-(perp**2) / 4.0)
        )

    dA = (g[1] - g[0]) ** 2
    return float(np.sum(psi(v1, a1, s1) * psi(v2, a2, s2)) * dA)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M, Dp = 5, 3
    V = rng.normal(size=(M, Dp))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    a = rng.uniform(-2.0, 2.0, size=M)
    s = 0.55
    K_tied_ref = kuu_tied_cip0028(V, a, s)
    K_tied_gen = kuu_heterogeneous(V, a, np.full(M, s))
    err_tied = np.max(np.abs(K_tied_ref - K_tied_gen))
    print(f"tied recovery max|Δ| = {err_tied:.3e}")
    assert err_tied < 1e-12

    # Diagonal must be 1.
    assert np.allclose(np.diag(K_tied_gen), 1.0, atol=1e-12)

    s_vec = np.array([0.4, 0.55, 0.7, 0.5, 0.6])
    K_het = kuu_heterogeneous(V, a, s_vec)
    assert np.allclose(np.diag(K_het), 1.0, atol=1e-10)
    print(f"heterogeneous diag max|1-K_ii| = {np.max(np.abs(1.0 - np.diag(K_het))):.3e}")

    # Quadrature gate in D'=2 for one heterogeneous pair.
    v1 = np.array([1.0, 0.0])
    v2 = np.array([np.cos(0.4), np.sin(0.4)])
    a1, a2 = 0.3, -0.7
    s1, s2 = 0.45, 0.8
    k_an = kuu_pair(v1, a1, s1, v2, a2, s2, Dp=2)
    k_q = _quad_1d(v1, a1, s1, v2, a2, s2)
    rel = abs(k_an - k_q) / max(abs(k_q), 1e-12)
    print(f"quad D'=2 analytic={k_an:.10f} quad={k_q:.10f} rel={rel:.3e}")
    assert rel < 2e-4
    print("ok")
