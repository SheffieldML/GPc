"""Variance captured by M isotropic weight-space windows vs input dimension D."""
import json
import numpy as np
from wsfeatures import WSFeatures, williams_kernel

rng = np.random.default_rng(3)
rows = []
print(" D   M    kappa  captured  cond(Kuu)")
for D in [1, 2, 4, 8, 16]:
    S = np.concatenate([np.full(D, 1.0 / D), [0.01]])
    for M in [64, 1024]:
        best = None
        for kappa in [0.5, 1.0, 1.5, 2.0, 3.0]:
            s = kappa * np.sqrt(2.0 / D)
            caps, conds = [], []
            for _ in range(3):
                C = rng.normal(size=(M, D + 1)) * np.sqrt(2 * S)
                f = WSFeatures(C, s, 1.0, S)
                Kuu = f.Kuu() + 1e-10 * np.eye(M)
                H = rng.normal(size=(20, D)) * np.sqrt(0.25)
                Kf = f.Kfu(H)
                L = np.linalg.cholesky(Kuu)
                q = np.sum(np.linalg.solve(L, Kf.T) ** 2, axis=0)
                k = np.diag(williams_kernel(H, H, 1.0, S))
                caps.append(np.mean(q / k))
                conds.append(np.linalg.cond(Kuu))
            row = (float(np.mean(caps)), kappa, float(np.median(conds)))
            if best is None or row[0] > best[0]:
                best = row
        rows.append(dict(D=D, M=M, captured=best[0], kappa=best[1], cond=best[2]))
        print(f"{D:2d} {M:5d}   {best[1]:4.1f}    {best[0]:.4f}   {best[2]:.1e}", flush=True)
json.dump(rows, open("dim_sweep.json", "w"), indent=1)
