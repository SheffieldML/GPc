"""Weight-space inducing features for the erf (Williams MLP) kernel.

Model: f(h) = sigma * int erf(w^T h~) sqrt(p(w)) dW(w),  p = N(0, Sigma),  h~ = (h, 1).
  => k(h, h') = sigma^2 E_p[erf(w^T h~) erf(w^T h~')]
             = sigma^2 (2/pi) asin( 2 h~^T Sigma h~' / sqrt((1 + 2 h~^T Sigma h~)(1 + 2 h~'^T Sigma h~')) ).
Inducing: u_m = int psi_m(w) dW(w), psi_m L2-normalised isotropic Gaussian window,
centre c_m, width s (fixed in w-space, so K_uu is kernel-free).
  K_uu[m, n] = exp(-||c_m - c_n||^2 / (4 s^2)).
  K_fu(h, m) = sigma * int erf(w^T h~) sqrt(p(w)) psi_m(w) dw
             = sigma * A_m * erf( mu_m^T h~ / sqrt(1 + 2 h~^T C h~) ),
  P = Sigma^{-1}/2 + I/s^2, C = P^{-1}, mu_m = C c_m / s^2,
  A_m = (pi s^2)^{-D'/4} (2 pi)^{-D'/4} |Sigma|^{-1/4} (2 pi)^{D'/2} |P|^{-1/2}
        * exp( mu_m^T P mu_m / 2 - ||c_m||^2 / (2 s^2) ).
The sqrt(p) factor is what makes K_fu depend on Sigma (weight / bias variances).
Sigma is diagonal throughout.
"""
import numpy as np
from scipy.special import erf

SQRT_PI = np.sqrt(np.pi)


def aug(h):
    """Append the bias coordinate: (..., D) -> (..., D+1)."""
    h = np.atleast_2d(h)
    return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)


def williams_kernel(H1, H2, sigma2, Sdiag):
    A1, A2 = aug(H1), aug(H2)
    num = 2.0 * (A1 * Sdiag) @ A2.T
    d1 = 1.0 + 2.0 * np.sum(A1 * A1 * Sdiag, axis=1)
    d2 = 1.0 + 2.0 * np.sum(A2 * A2 * Sdiag, axis=1)
    return sigma2 * (2.0 / np.pi) * np.arcsin(num / np.sqrt(np.outer(d1, d2)))


class WSFeatures:
    def __init__(self, centres, s, sigma2, Sdiag):
        self.c = np.asarray(centres, float)          # M x D'
        self.s = float(s)
        self.sigma = np.sqrt(sigma2)
        self.S = np.asarray(Sdiag, float)            # D'
        Dp = self.c.shape[1]
        Pdiag = 0.5 / self.S + 1.0 / self.s ** 2
        self.Cdiag = 1.0 / Pdiag
        self.mu = self.c * (self.Cdiag / self.s ** 2)  # M x D'
        logA = (-Dp / 4.0 * np.log(np.pi * self.s ** 2)
                - Dp / 4.0 * np.log(2 * np.pi)
                - 0.25 * np.sum(np.log(self.S))
                + Dp / 2.0 * np.log(2 * np.pi)
                - 0.5 * np.sum(np.log(Pdiag))
                + 0.5 * np.sum(self.mu ** 2 * Pdiag, axis=1)
                - 0.5 * np.sum(self.c ** 2, axis=1) / self.s ** 2)
        self.logA = logA
        self.A = np.exp(logA)

    def Kuu(self):
        d2 = np.sum((self.c[:, None, :] - self.c[None, :, :]) ** 2, axis=2)
        return np.exp(-d2 / (4.0 * self.s ** 2))

    def Kfu(self, H):
        X = aug(H)                                          # N x D'
        r = np.sqrt(1.0 + 2.0 * np.sum(X * X * self.Cdiag, axis=1))  # N
        g = X @ self.mu.T                                   # N x M
        return self.sigma * self.A[None, :] * erf(g / r[:, None])

    def Kfu_jac(self, h):
        """d Kfu(h, m) / d h for a single point h (D,) -> M x D."""
        x = aug(h)[0]
        r2 = 1.0 + 2.0 * np.sum(x * x * self.Cdiag)
        r = np.sqrt(r2)
        g = self.mu @ x                                     # M
        dr = (2.0 * self.Cdiag * x) / r                     # D'
        dz = self.mu / r - np.outer(g / r2, dr)             # M x D'
        ep = (2.0 / SQRT_PI) * np.exp(-(g / r) ** 2)        # erf'
        J = self.sigma * (self.A * ep)[:, None] * dz        # M x D'
        return J[:, :-1]


def kfu_by_quadrature_1d(h, centre, s, sigma2, Sdiag, n=801, span=8.0):
    """Direct 2-D quadrature of int erf(w^T h~) sqrt(p(w)) psi(w) dw for D = 1 (D' = 2)."""
    sd = np.sqrt(2.0 * Sdiag) * span + np.abs(centre) + span * s
    g0 = np.linspace(-sd[0], sd[0], n)
    g1 = np.linspace(-sd[1], sd[1], n)
    W0, W1 = np.meshgrid(g0, g1, indexing="ij")
    x = np.array([h, 1.0])
    phi = erf(W0 * x[0] + W1 * x[1])
    logsqrtp = (-0.5 * np.log(2 * np.pi) - 0.25 * np.sum(np.log(Sdiag))
                - (W0 ** 2 / Sdiag[0] + W1 ** 2 / Sdiag[1]) / 4.0)
    logpsi = (-0.5 * np.log(np.pi * s ** 2)
              - ((W0 - centre[0]) ** 2 + (W1 - centre[1]) ** 2) / (2 * s ** 2))
    integrand = phi * np.exp(logsqrtp + logpsi)
    dw = (g0[1] - g0[0]) * (g1[1] - g1[0])
    return np.sqrt(sigma2) * integrand.sum() * dw
