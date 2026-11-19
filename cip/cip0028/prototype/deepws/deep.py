"""Part 2/3: end-to-end Jacobian spectra of deep (weight-space) GP layer stacks.

Every layer maps R^D -> R^D. We propagate inputs, take per-layer Jacobians and
accumulate the log singular-value spectrum of J_L ... J_1 with the QR
(Lyapunov) method; it is validated against a direct SVD at small depth.

Layer families
  rf_erf / rf_relu : exact infinite-width GP layer (Williams erf / ReLU arc-cos)
                     approximated with K random features; output weights A are a
                     GP prior sample (Gaussian, independent of W).
  tied             : "posterior-mean path" with output weights tied to the hidden
                     weights, A = c W_x^T, W_x with orthonormal columns.
  residual         : h + f(h) with an rf_erf f.
  ws               : CIP-0028 weight-space sparse layer, f(h) = K_fu(h) L^{-T} V,
                     V ~ N(0, I), M windows of width s.
Criticality is set by bisection on the output scale so the mean squared singular
value of a single layer's Jacobian at the stationary input scale is 1 (chi = 1).
"""
import json
import sys
import numpy as np
import mpmath as mp

mp.mp.dps = 160
from scipy.special import erf
from wsfeatures import WSFeatures

D = 16
K = 512
DEPTHS = [1, 2, 4, 8, 16, 32, 64]
LMAX = DEPTHS[-1]
N_IN = 6
SEEDS = 3
BURN = 15
SW2 = 1.0      # total weight variance (Sigma_x = SW2 / D * I)
SB2 = 0.01     # bias variance

erf_p = lambda z: (2.0 / np.sqrt(np.pi)) * np.exp(-z * z)
relu = lambda z: np.maximum(z, 0.0)
relu_p = lambda z: (z > 0).astype(float)


def orth_cols(rng, n, d):
    q, r = np.linalg.qr(rng.normal(size=(n, d)))
    return q * np.sign(np.diag(r))


# ---------------------------------------------------------------- layers
class RFLayer:
    def __init__(self, rng, act, sig2, ortho=False, tied=False, sw2=SW2, sb2=SB2):
        self.act, self.actp = (erf, erf_p) if act == "erf" else (relu, relu_p)
        if ortho or tied:
            Wx = orth_cols(rng, K, D) * np.sqrt(K * sw2 / D)
        else:
            Wx = rng.normal(size=(K, D)) * np.sqrt(sw2 / D)
        self.Wx = Wx
        self.b = rng.normal(size=K) * np.sqrt(sb2)
        if tied:
            self.A = Wx.T / np.sqrt(sw2 / D)        # ~ orthonormal rows * sqrt(K)
        else:
            self.A = rng.normal(size=(D, K))
        self.scale = np.sqrt(sig2 / K)

    def __call__(self, h):
        z = self.Wx @ h + self.b
        return self.scale * self.A @ self.act(z), self.scale * (self.A * self.actp(z)) @ self.Wx


class ResidualLayer:
    def __init__(self, rng, sig2):
        self.f = RFLayer(rng, "erf", sig2)

    def __call__(self, h):
        y, J = self.f(h)
        return h + y, np.eye(D) + J


class WSLayer:
    def __init__(self, rng, sig2, M, kappa, ortho=False):
        Sdiag = np.concatenate([np.full(D, SW2 / D), [SB2]])
        if ortho:
            cx = orth_cols(rng, M, D) * np.sqrt(M * 2 * SW2 / D)
        else:
            cx = rng.normal(size=(M, D)) * np.sqrt(2 * SW2 / D)
        cb = rng.normal(size=(M, 1)) * np.sqrt(2 * SB2)
        s = kappa * np.sqrt(2 * SW2 / D)
        self.feat = WSFeatures(np.hstack([cx, cb]), s, sig2, Sdiag)
        Kuu = self.feat.Kuu()
        self.cond = np.linalg.cond(Kuu)
        Lc = np.linalg.cholesky(Kuu + 1e-10 * np.eye(M))
        V = rng.normal(size=(M, D))
        self.B = np.linalg.solve(Lc.T, V)          # L^{-T} V, M x D
        self.Lc = Lc
        self.Sdiag = Sdiag
        self.sig2 = sig2

    def __call__(self, h):
        y = (self.feat.Kfu(h[None, :]) @ self.B)[0]
        J = self.B.T @ self.feat.Kfu_jac(h)
        return y, J

    def captured(self, h):
        """Q(h,h) / k(h,h): prior variance captured by the M windows."""
        from wsfeatures import williams_kernel
        kf = self.feat.Kfu(h[None, :])[0]
        q = np.sum(np.linalg.solve(self.Lc, kf) ** 2)
        k = williams_kernel(h[None, :], h[None, :], self.sig2, self.Sdiag)[0, 0]
        return q / k


# ---------------------------------------------------------------- measurement
def lyapunov_run(make_layer, rng, depths=DEPTHS, detnorm=False, check_svd=False):
    """Mean over inputs of log singular values of the product Jacobian at each depth."""
    out = {L: [] for L in depths}
    svd_err = []
    for _ in range(N_IN):
        h = rng.normal(size=D)
        for _ in range(BURN):
            h, _ = make_layer(rng)(h)
        Q = np.eye(D)
        logs = np.zeros(D)
        P = np.eye(D)
        Pm = mp.matrix(np.eye(D).tolist())
        for l in range(1, depths[-1] + 1):
            layer = make_layer(rng)
            h, J = layer(h)
            if detnorm:
                sgn, ld = np.linalg.slogdet(J)
                J = J / np.exp(ld / D)
            Z = J @ Q
            Q, R = np.linalg.qr(Z)
            d = np.diag(R)
            Q = Q * np.sign(d)
            logs = logs + np.log(np.abs(d))
            Pm = mp.matrix(J.tolist()) * Pm
            if l in out:
                sv = mp.svd_r(Pm, compute_uv=False)
                exact = np.sort(np.array([float(mp.log(x)) for x in sv]))
                if check_svd:
                    svd_err.append(np.max(np.abs(np.sort(logs) - exact)))
                out[l].append(exact)
    res = {L: np.array(v) for L, v in out.items()}
    return res, (np.nanmax(svd_err) if svd_err else None)


def per_layer_chi(make_layer, rng, n=40):
    """Mean squared singular value of one layer's Jacobian at the stationary scale."""
    vals, qs = [], []
    for _ in range(n // 8 + 1):
        h = rng.normal(size=D)
        for _ in range(BURN):
            h, _ = make_layer(rng)(h)
        for _ in range(8):
            layer = make_layer(rng)
            h2, J = layer(h)
            vals.append(np.sum(J * J) / D)
            qs.append(np.mean(h * h))
            h = h2
    return float(np.mean(vals)), float(np.mean(qs))


def tune_scale(factory, rng_seed, target=1.0, lo=0.05, hi=50.0, iters=22):
    """Bisection (in log) on sig2 so that per-layer chi == target."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        chi, _ = per_layer_chi(lambda r: factory(r, mid), np.random.default_rng(rng_seed))
        if chi > target:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def summarise(res):
    rows = {}
    for L, arr in res.items():
        rows[L] = dict(mean=float(arr.mean()), spread=float(arr.std(axis=1).mean()),
                       range=float((arr[:, -1] - arr[:, 0]).mean()))
    return rows


def run_condition(name, factory, sig2, detnorm=False, check=False):
    allres = []
    errs = []
    for sd in range(SEEDS):
        rng = np.random.default_rng(1000 + sd)
        res, err = lyapunov_run(lambda r: factory(r, sig2), rng, detnorm=detnorm, check_svd=check)
        allres.append(res)
        if err is not None:
            errs.append(err)
    merged = {L: np.concatenate([r[L] for r in allres]) for L in DEPTHS}
    s = summarise(merged)
    return s, (max(errs) if errs else None)


def main():
    results = {}
    log = []

    def report(name, s, extra=""):
        line = f"{name:44s} " + "  ".join(
            f"L={L:>2d}: mean {s[L]['mean']:+6.2f} sd {s[L]['spread']:5.2f}" for L in [8, 32, 64]) + extra
        print(line, flush=True)
        log.append(line)

    fac = {
        "rf_erf_gauss": lambda r, s2: RFLayer(r, "erf", s2),
        "rf_erf_orth": lambda r, s2: RFLayer(r, "erf", s2, ortho=True),
        "rf_relu_gauss": lambda r, s2: RFLayer(r, "relu", s2),
        "tied_erf": lambda r, s2: RFLayer(r, "erf", s2, tied=True),
        "tied_relu": lambda r, s2: RFLayer(r, "relu", s2, tied=True),
    }
    scales = {}
    print("== Criticality tuning (per-layer chi = 1) ==", flush=True)
    for k, f in fac.items():
        scales[k] = tune_scale(f, 7)
        chi, q = per_layer_chi(lambda r: f(r, scales[k]), np.random.default_rng(99), n=80)
        print(f"  {k:16s} sig2*={scales[k]:.4f}  check chi={chi:.3f}  q*={q:.4f}", flush=True)
    # Analytic chi for the exact erf layer at the tuned scale.
    s2 = scales["rf_erf_gauss"]
    q = 1.0
    for _ in range(500):
        v = SW2 * q + SB2
        q = s2 * (2 / np.pi) * np.arcsin(2 * v / (1 + 2 * v))
    v = SW2 * q + SB2
    chi_an = s2 * SW2 * (4 / np.pi) / np.sqrt(1 + 4 * v)
    print(f"  analytic chi for rf_erf at sig2*: {chi_an:.3f} (q* = {q:.4f})", flush=True)

    print("\n== Spectra of log singular values of J_L...J_1 (mean, within-spectrum sd) ==", flush=True)
    s, err = run_condition("rf_erf_gauss", fac["rf_erf_gauss"], scales["rf_erf_gauss"], check=True)
    results["GP prior, erf, Gaussian W"] = s
    report("GP prior, erf, Gaussian W (critical)", s, f"   [max |Lyapunov-QR - exact| over depths: {err:.1f}]")
    for name, mult in [("GP prior, erf, Gaussian W (ordered 0.8)", 0.8), ("GP prior, erf, Gaussian W (chaotic 1.25)", 1.25)]:
        sc = tune_scale(fac["rf_erf_gauss"], 7, target=mult)
        s, _ = run_condition(name, fac["rf_erf_gauss"], sc)
        results[name] = s
        report(name, s)
    s, _ = run_condition("orth", fac["rf_erf_orth"], scales["rf_erf_orth"])
    results["GP prior, erf, orthogonal W"] = s
    report("GP prior, erf, orthogonal W (critical)", s)
    s, _ = run_condition("relu", fac["rf_relu_gauss"], scales["rf_relu_gauss"])
    results["GP prior, ReLU, Gaussian W"] = s
    report("GP prior, ReLU, Gaussian W (critical)", s)
    s, _ = run_condition("detnorm", fac["rf_erf_gauss"], scales["rf_erf_gauss"], detnorm=True)
    results["GP prior, erf, det-normalised (|det J_l| = 1)"] = s
    report("GP prior, erf, det-normalised |det J_l|=1", s)
    s, _ = run_condition("tied_erf", fac["tied_erf"], scales["tied_erf"])
    results["Mean path, tied orthogonal, erf"] = s
    report("Mean path, tied orthogonal, erf (critical)", s)
    s, _ = run_condition("tied_relu", fac["tied_relu"], scales["tied_relu"])
    results["Mean path, tied orthogonal, ReLU"] = s
    report("Mean path, tied orthogonal, ReLU (critical)", s)
    # Pennington regime: isometry for erf needs q* -> 0, i.e. small bias variance.
    for sb2 in [1e-4, 1e-6]:
        f = lambda r, s2, sb2=sb2: RFLayer(r, "erf", s2, tied=True, sb2=sb2)
        sc = tune_scale(f, 7)
        chi, q = per_layer_chi(lambda r: f(r, sc), np.random.default_rng(99), n=80)
        s, _ = run_condition("tied_erf_sb", f, sc)
        nm = f"Mean path, tied orthogonal, erf, sb2={sb2:g}"
        results[nm] = s
        report(nm + f" (q*={q:.3g})", s)

    # Residual: per-layer chi of f set to 1/LMAX (scaled) or 1 (unscaled).
    for name, tgt in [("Residual GP, chi_f = 1/L_max", 1.0 / LMAX), ("Residual GP, chi_f = 1", 1.0)]:
        sc = tune_scale(fac["rf_erf_gauss"], 7, target=tgt, lo=1e-5)
        s, _ = run_condition(name, lambda r, s2: ResidualLayer(r, s2), sc)
        results[name] = s
        report(name, s)

    json.dump({"scales": scales, "results": {k: {str(L): v for L, v in r.items()} for k, r in results.items()}},
              open("deep_results.json", "w"), indent=1)
    open("deep_log.txt", "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
