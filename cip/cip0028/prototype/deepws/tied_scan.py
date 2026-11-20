"""Scan the near-linear branch of the tied erf mean path: gain g^2 = sig2 / sig2_lin."""
import json
import numpy as np
from deep import RFLayer, per_layer_chi, run_condition, D, K, SW2

lin = D / (K * SW2) * np.pi / 4
rows = []
for g2 in [1.002, 1.01, 1.05, 1.2, 1.5]:
    f = lambda r, s2: RFLayer(r, "erf", s2, tied=True, sb2=1e-6)
    sc = lin * g2
    chi, q = per_layer_chi(lambda r: f(r, sc), np.random.default_rng(99), n=80)
    s, _ = run_condition("tied", f, sc)
    print(f"g^2={g2:6.3f}: chi={chi:.4f} q*={q:.4f} | "
          + "  ".join(f"L={L}: mean {s[L]['mean']:+.3f} sd {s[L]['spread']:.3f}" for L in [8, 32, 64]), flush=True)
    rows.append({"g2": g2, "chi": chi, "q": q, "res": {str(L): v for L, v in s.items()}})
# ReLU comparison at its tuned scale, for the same table.
json.dump(rows, open("tied_scan.json", "w"), indent=1)
