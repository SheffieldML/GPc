"""Rerun the tied (posterior-mean path) erf case on the near-linear critical branch."""
import numpy as np
import deep
from deep import RFLayer, tune_scale, per_layer_chi, run_condition, D, K, SW2

lin = D / (K * SW2) * np.pi / 4          # sig2 at which the linear-regime gain is exactly 1
print(f"linear-regime unit-gain sig2 = {lin:.5f}")
rows = []
for sb2 in [1e-2, 1e-4, 1e-6]:
    f = lambda r, s2, sb2=sb2: RFLayer(r, "erf", s2, tied=True, sb2=sb2)
    sc = tune_scale(f, 7, lo=lin * 0.999, hi=lin * 1.5)
    chi, q = per_layer_chi(lambda r: f(r, sc), np.random.default_rng(99), n=80)
    s, _ = run_condition("tied", f, sc)
    line = (f"Mean path, tied orth, erf, near-linear branch sb2={sb2:g}: sig2={sc:.5f} chi={chi:.3f} q*={q:.3g} | "
            + "  ".join(f"L={L}: mean {s[L]['mean']:+.2f} sd {s[L]['spread']:.2f}" for L in [8, 32, 64]))
    print(line, flush=True)
    rows.append((sb2, sc, chi, q, s))
import json
json.dump([{"sb2": r[0], "sig2": r[1], "chi": r[2], "q": r[3], "res": {str(L): v for L, v in r[4].items()}} for r in rows],
          open("tied_fix.json", "w"), indent=1)
