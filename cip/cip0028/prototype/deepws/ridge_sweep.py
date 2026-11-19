"""Variance captured vs dimension: ridge windows (whitened) vs isotropic windows."""
import json
import numpy as np
from ridge import RidgeFeatures
from wsfeatures import williams_kernel

rng = np.random.default_rng(3)
iso = {(r["D"], r["M"]): r for r in json.load(open("dim_sweep.json"))}
rows = []
print(" D     M   n_off    s   ridge-captured  cond(Kuu)  | isotropic-captured")
for D in [1, 2, 4, 8, 16, 32]:
    S = np.concatenate([np.full(D, 1.0 / D), [0.01]])
    for M in [64, 1024]:
        best = None
        for n_off in [2, 4, 8]:
            ndir = M // n_off
            offs = np.linspace(-2.5, 2.5, n_off)
            for s in [0.3, 0.5, 0.8, 1.2]:
                caps, conds = [], []
                for _ in range(3):
                    V = rng.normal(size=(ndir, D + 1))
                    V = np.repeat(V, n_off, axis=0)
                    a = np.tile(offs, ndir)
                    f = RidgeFeatures(V, a, s, 1.0, S)
                    Kuu = f.Kuu()
                    H = rng.normal(size=(20, D)) * np.sqrt(0.25)
                    Kf = f.Kfu(H)
                    L = np.linalg.cholesky(Kuu + 1e-10 * np.eye(M))
                    q = np.sum(np.linalg.solve(L, Kf.T) ** 2, axis=0)
                    k = np.diag(williams_kernel(H, H, 1.0, S))
                    caps.append(np.mean(q / k))
                    conds.append(np.linalg.cond(Kuu))
                row = (float(np.mean(caps)), n_off, s, float(np.median(conds)))
                if best is None or row[0] > best[0]:
                    best = row
        ic = iso.get((D, M), {}).get("captured", float("nan"))
        rows.append(dict(D=D, M=M, captured=best[0], n_off=best[1], s=best[2], cond=best[3], iso=ic))
        print(f"{D:2d}  {M:5d}   {best[1]:3d}   {best[2]:4.1f}      {best[0]:.4f}      {best[3]:.1e}  |   {ic:.4f}", flush=True)
json.dump(rows, open("ridge_sweep.json", "w"), indent=1)
