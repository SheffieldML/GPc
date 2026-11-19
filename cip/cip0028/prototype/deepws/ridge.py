"""Ridge (directional) weight-space windows in prior-whitened coordinates.

Whitening: omega = Sigma^{-1/2} w, x^ = Sigma^{1/2} x~ (x~ = (x, 1)). With
u_m = int psi_m(Sigma^{-1/2} w) |Sigma|^{-1/4} dW(w), K_uu = <psi_m, psi_n>_{L2(omega)}
is kernel-free and all kernel dependence sits in x^.

Ridge window (unit direction v_m, offset a_m, width s):
  psi_m(omega) = g_m(v_m^T omega) * sqrt(N(Pi_m^perp omega; 0, I)),
  g_m(t) = (pi s^2)^{-1/4} exp(-(t - a_m)^2 / (2 s^2)).
Along v_m: a 1-D window; across: exactly sqrt of the whitened prior, so the
complement overlaps the prior perfectly and nothing decays with dimension.

  K_fu(x, m) = sigma * A_m * erf( mu_m v_m^T x^ / sqrt(1 + 2 (||x^||^2 - (1 - c_s) (v_m^T x^)^2)) ),
  c_s = 2 s^2 / (2 + s^2), mu_m = a_m c_s / s^2,
  A_m = int sqrt(N(t; 0, 1)) g_m(t) dt
      = (2 pi)^{-1/4} (pi s^2)^{-1/4} sqrt(2 pi c_s) exp(mu_m^2 / (2 c_s) - a_m^2 / (2 s^2)).
  K_uu[m, n]: Gaussian integral with precision P = I + alpha (v_m v_m^T + v_n v_n^T),
  alpha = 1/s^2 - 1/2, reduced to 2 x 2 algebra (Sylvester / Woodbury).
"""
import numpy as np
from scipy.special import erf


class RidgeFeatures:
    def __init__(self, V, a, s, sigma2, Sdiag):
        self.V = V / np.linalg.norm(V, axis=1, keepdims=True)   # M x D'
        self.a = np.asarray(a, float)
        self.s = float(s)
        self.sigma = np.sqrt(sigma2)
        self.sqS = np.sqrt(np.asarray(Sdiag, float))
        s2 = self.s ** 2
        self.cs = 2 * s2 / (2 + s2)
        self.mu = self.a * self.cs / s2
        self.A = ((2 * np.pi) ** -0.25 * (np.pi * s2) ** -0.25 * np.sqrt(2 * np.pi * self.cs)
                  * np.exp(self.mu ** 2 / (2 * self.cs) - self.a ** 2 / (2 * s2)))

    def xhat(self, H):
        H = np.atleast_2d(H)
        return np.concatenate([H, np.ones((H.shape[0], 1))], axis=1) * self.sqS

    def Kfu(self, H):
        X = self.xhat(H)
        t = X @ self.V.T                                  # N x M, v^T x^
        n2 = np.sum(X * X, axis=1)[:, None]
        den = np.sqrt(1 + 2 * (n2 - (1 - self.cs) * t * t))
        return self.sigma * self.A[None, :] * erf(self.mu[None, :] * t / den)

    def Kuu(self):
        M, Dp = self.V.shape
        s2 = self.s ** 2
        alpha = 1.0 / s2 - 0.5
        G12 = self.V @ self.V.T                           # cosines
        # log normaliser of each psi: (pi s^2)^{-1/4} (2 pi)^{-(D'-1)/4}; product of two -> squared.
        lognorm2 = -0.5 * np.log(np.pi * s2) - (Dp - 1) / 2.0 * np.log(2 * np.pi)
        am, an = self.a[:, None], self.a[None, :]
        # 2x2 Gram G = [[1, c], [c, 1]]; det(I2 + alpha G); beta = (a_m, a_n)/s^2.
        c = G12
        detI = (1 + alpha) ** 2 - (alpha * c) ** 2
        # P^{-1} restricted: b^T P^{-1} b = beta^T (G - G (alpha^{-1} I + G)^{-1} G) beta = beta^T H beta,
        # with H = G (I + alpha G)^{-1}.
        # (I + alpha G)^{-1} = 1/detI * [[1+alpha, -alpha c], [-alpha c, 1+alpha]]
        inv11 = (1 + alpha) / detI
        inv12 = -alpha * c / detI
        H11 = inv11 + c * inv12
        H12 = inv12 + c * inv11
        bm, bn = am / s2, an / s2
        quad = H11 * (bm ** 2 + bn ** 2) + 2 * H12 * bm * bn
        const = -(am ** 2 + bn * 0 + an ** 2) / (2 * s2)
        logK = (lognorm2 + Dp / 2.0 * np.log(2 * np.pi) - 0.5 * np.log(detI)
                + 0.5 * quad + const)
        K = np.exp(logK)
        return 0.5 * (K + K.T)


def ridge_kfu_mc(x, v, a, s, Sdiag, sigma2, n=400000, rng=None):
    """Monte Carlo of sigma int erf(w^T x~) sqrt(p(w)) psi(Sigma^{-1/2} w) |Sigma|^{-1/4} dw, in omega."""
    rng = rng or np.random.default_rng(0)
    Dp = len(v)
    xh = np.append(x, 1.0) * np.sqrt(Sdiag)
    v = v / np.linalg.norm(v)
    # Sample omega ~ N(0, I) across, and t along v from the (unnormalised) Gaussian sqrt(N(t))g(t).
    cs = 2 * s * s / (2 + s * s)
    mu = a * cs / (s * s)
    t = mu + np.sqrt(cs) * rng.normal(size=n)
    z = rng.normal(size=(n, Dp))
    z = z - np.outer(z @ v, v)
    om = z + np.outer(t, v)
    A = ((2 * np.pi) ** -0.25 * (np.pi * s * s) ** -0.25 * np.sqrt(2 * np.pi * cs)
         * np.exp(mu ** 2 / (2 * cs) - a ** 2 / (2 * s * s)))
    return np.sqrt(sigma2) * A * np.mean(erf(om @ xh))


def ridge_kuu_quad_1d(v1, a1, v2, a2, s, n=1201, span=9.0):
    """Direct 2-D quadrature of <psi_1, psi_2> for D' = 2."""
    g = np.linspace(-span, span, n)
    W0, W1 = np.meshgrid(g, g, indexing="ij")
    def psi(v, a):
        v = v / np.linalg.norm(v)
        t = W0 * v[0] + W1 * v[1]
        perp = -W0 * v[1] + W1 * v[0]
        return ((np.pi * s * s) ** -0.25 * np.exp(-(t - a) ** 2 / (2 * s * s))
                * (2 * np.pi) ** -0.25 * np.exp(-perp ** 2 / 4))
    dA = (g[1] - g[0]) ** 2
    return np.sum(psi(v1, a1) * psi(v2, a2)) * dA
