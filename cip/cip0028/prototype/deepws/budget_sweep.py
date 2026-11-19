"""Best variance captured subject to cond(K_uu) <= 1e4: ridge vs isotropic windows."""
import json
import numpy as np
from ridge import RidgeFeatures
from wsfeatures import WSFeatures, williams_kernel

BUDGET = 1e4
rng = np.random.default_rng(11)


def capture(f, M, D, S):
    Kuu = f.Kuu()
    cond = np.linalg.cond(Kuu)
    H = rng.normal(size=(20, D)) * np.sqrt(0.25)
    Kf = f.Kfu(H)
    L = np.linalg.cholesky(Kuu + 1e-10 * np.eye(M))
    q = np.sum(np.linalg.solve(L, Kf.T) ** 2, axis=0)
    k = np.diag(williams_kernel(H, H, 1.0, S))
    return float(np.mean(q / k)), float(cond)


rows = []
print(f" D     M   ridge(cond<=1e4)   isotropic(cond<=1e4)")
for D in [1, 2, 4, 8, 16, 32]:
    S = np.concatenate([np.full(D, 1.0 / D), [0.01]])
    for M in [64, 1024]:
        best_r, best_i = (0.0, None), (0.0, None)
        for n_off in [2, 4, 8]:
            for s in [0.2, 0.3, 0.5, 0.8, 1.2]:
                vals = []
                for _ in range(2):
                    V = np.repeat(rng.normal(size=(M // n_off, D + 1)), n_off, axis=0)
                    a = np.tile(np.linspace(-2.5, 2.5, n_off), M // n_off)
                    vals.append(capture(RidgeFeatures(V, a, s, 1.0, S), M, D, S))
                cap, cond = np.mean([v[0] for v in vals]), np.max([v[1] for v in vals])
                if cond <= BUDGET and cap > best_r[0]:
                    best_r = (cap, dict(n_off=n_off, s=s, cond=cond))
        for kappa in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            vals = []
            for _ in range(2):
                C = rng.normal(size=(M, D + 1)) * np.sqrt(2 * S)
                vals.append(capture(WSFeatures(C, kappa * np.sqrt(2.0 / D), 1.0, S), M, D, S))
            cap, cond = np.mean([v[0] for v in vals]), np.max([v[1] for v in vals])
            if cond <= BUDGET and cap > best_i[0]:
                best_i = (cap, dict(kappa=kappa, cond=cond))
        rows.append(dict(D=D, M=M, ridge=best_r[0], ridge_cfg=best_r[1], iso=best_i[0], iso_cfg=best_i[1]))
        print(f"{D:2d}  {M:5d}      {best_r[0]:.4f} {best_r[1]}      {best_i[0]:.4f} {best_i[1]}", flush=True)
json.dump(rows, open("budget_sweep.json", "w"), indent=1)
