"""Part 3: CIP-0028 weight-space sparse layers in depth.

(a) At the exact erf layer's critical hyperparameters, how do M and the window
    width s = kappa * sqrt(2 SW2 / D) change the per-layer chi, the prior variance
    captured Q(h,h)/k(h,h), and cond(K_uu)?
(b) Re-tune the output scale for the sparse layer itself and measure end-to-end
    spectra: Gaussian vs orthogonal centres, and a residual version.
"""
import json
import numpy as np
from deep import (WSLayer, RFLayer, per_layer_chi, tune_scale, run_condition, D, BURN)

SIG2_EXACT = 1.1354   # tuned critical sig2 of the exact erf GP layer (deep.py)

out = {"a": [], "b": {}}
print("== (a) per-layer statistics at the exact layer's critical sig2 ==", flush=True)
print("   M  kappa   chi    var-captured   median cond(Kuu)", flush=True)
for M in [64, 256]:
    for kappa in [0.25, 0.5, 1.0, 2.0]:
        rng = np.random.default_rng(5)
        f = lambda r: WSLayer(r, SIG2_EXACT, M, kappa)
        chi, q = per_layer_chi(f, rng, n=24)
        caps, conds = [], []
        for _ in range(6):
            h = rng.normal(size=D)
            for _ in range(BURN):
                h, _ = f(rng)(h)
            lay = f(rng)
            caps.append(lay.captured(h))
            conds.append(lay.cond)
        row = dict(M=M, kappa=kappa, chi=chi, q=q, captured=float(np.mean(caps)), cond=float(np.median(conds)))
        out["a"].append(row)
        print(f"  {M:4d}  {kappa:4.2f}  {chi:6.3f}     {row['captured']:.3f}         {row['cond']:.1e}", flush=True)

print("\n== (b) sparse layers re-tuned to chi = 1, M = 256 ==", flush=True)
M = 256
for kappa in [0.5, 1.0]:
    for name, ortho in [("Gaussian centres", False), ("orthogonal centres", True)]:
        fac = lambda r, s2, kappa=kappa, ortho=ortho: WSLayer(r, s2, M, kappa, ortho=ortho)
        sc = tune_scale(fac, 7, lo=0.05, hi=5000.0, iters=24)
        chi, q = per_layer_chi(lambda r: fac(r, sc), np.random.default_rng(99), n=40)
        s, _ = run_condition("ws", fac, sc)
        key = f"WS prior, M={M}, kappa={kappa}, {name}"
        out["b"][key] = dict(sig2=sc, shift=sc / SIG2_EXACT, chi=chi, q=q, res={str(L): v for L, v in s.items()})
        print(f"{key:48s} sig2*={sc:9.3f} (x{sc / SIG2_EXACT:6.2f} exact) chi={chi:.3f} q*={q:.3f} | "
              + "  ".join(f"L={L}: mean {s[L]['mean']:+.2f} sd {s[L]['spread']:.2f}" for L in [8, 32, 64]), flush=True)

# Residual sparse stack: f scaled so chi_f = 1/64.
kappa = 0.5


class WSResidual:
    def __init__(self, r, s2):
        self.f = WSLayer(r, s2, M, kappa)

    def __call__(self, h):
        y, J = self.f(h)
        return h + y, np.eye(D) + J


fac = lambda r, s2: WSLayer(r, s2, M, kappa)
sc = tune_scale(fac, 7, target=1.0 / 64, lo=1e-4, hi=5000.0, iters=26)
s, _ = run_condition("wsres", lambda r, s2: WSResidual(r, s2), sc)
key = f"WS residual, M={M}, kappa={kappa}, chi_f=1/64"
out["b"][key] = dict(sig2=sc, res={str(L): v for L, v in s.items()})
print(f"{key:48s} sig2={sc:9.4f} | "
      + "  ".join(f"L={L}: mean {s[L]['mean']:+.2f} sd {s[L]['spread']:.2f}" for L in [8, 32, 64]), flush=True)
json.dump(out, open("ws_results.json", "w"), indent=1)
