"""Part 1: validate the weight-space K_fu formula (with the sqrt(p) factor)."""
import numpy as np
from wsfeatures import WSFeatures, kfu_by_quadrature_1d, williams_kernel

rng = np.random.default_rng(0)

print("== 1a. Closed-form K_fu vs direct quadrature (D = 1) ==")
worst = 0.0
for trial in range(12):
    sw2 = rng.uniform(0.3, 3.0)
    sb2 = rng.uniform(0.05, 1.0)
    S = np.array([sw2, sb2])
    s = rng.uniform(0.2, 1.0)
    c = rng.normal(size=2) * np.sqrt(2 * S)
    h = rng.uniform(-2, 2)
    f = WSFeatures(c[None, :], s, 1.3, S)
    cf = f.Kfu(np.array([[h]]))[0, 0]
    q = kfu_by_quadrature_1d(h, c, s, 1.3, S)
    rel = abs(cf - q) / max(abs(q), 1e-12)
    worst = max(worst, rel)
    print(f"  sw2={sw2:.2f} sb2={sb2:.2f} s={s:.2f} h={h:+.2f}: closed={cf:+.6e} quad={q:+.6e} rel={rel:.1e}")
print(f"  worst relative error: {worst:.1e}")

print("\n== 1b. Jacobian of K_fu vs finite differences (D = 4) ==")
D = 4
S = np.concatenate([np.full(D, 0.8 / D), [0.1]])
c = rng.normal(size=(6, D + 1)) * np.sqrt(2 * S)
f = WSFeatures(c, 0.3, 1.0, S)
h = rng.normal(size=D)
J = f.Kfu_jac(h)
eps = 1e-6
Jfd = np.zeros_like(J)
for j in range(D):
    e = np.zeros(D); e[j] = eps
    Jfd[:, j] = (f.Kfu(h + e)[0] - f.Kfu(h - e)[0]) / (2 * eps)
print(f"  max |J - J_fd| = {np.max(np.abs(J - Jfd)):.1e} (max |J| = {np.max(np.abs(J)):.2e})")

print("\n== 1c. K_fu depends on weight / bias variance only through sqrt(p) ==")
c1 = np.array([[0.7, -0.2]])
for sw2 in [0.25, 1.0, 4.0]:
    S = np.array([sw2, 0.2])
    f = WSFeatures(c1, 0.4, 1.0, S)
    print(f"  sw2={sw2:4.2f}: K_fu(h=1) = {f.Kfu(np.array([[1.0]]))[0,0]:+.5f}   K_uu = {f.Kuu()[0,0]:.3f}")

print("\n== 1d. Nystrom Q = K_fu K_uu^-1 K_uf -> Williams kernel as windows densify (D = 1) ==")
S = np.array([1.0, 0.2])
H = np.linspace(-3, 3, 41)[:, None]
K = williams_kernel(H, H, 1.0, S)
for n_side, s in [(5, 0.6), (9, 0.35), (15, 0.22), (25, 0.13)]:
    # Grid of centres covering +-4 sd of the sqrt(p) Gaussian N(0, 2 Sigma).
    g0 = np.linspace(-4, 4, n_side) * np.sqrt(2 * S[0])
    g1 = np.linspace(-4, 4, n_side) * np.sqrt(2 * S[1])
    C = np.array([[a, b] for a in g0 for b in g1])
    f = WSFeatures(C, s, 1.0, S)
    Kuu = f.Kuu() + 1e-10 * np.eye(len(C))
    Kf = f.Kfu(H)
    Lc = np.linalg.cholesky(Kuu)
    B = np.linalg.solve(Lc, Kf.T)
    Q = B.T @ B
    print(f"  M={len(C):4d} s={s:.2f}: max|K-Q|/max K = {np.max(np.abs(K - Q)) / K.max():.2e}, "
          f"min eig(K-Q) = {np.linalg.eigvalsh(K - Q).min():+.1e}, diag captured = {np.mean(np.diag(Q)/np.diag(K)):.4f}")
