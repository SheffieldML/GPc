# CIP-0035 validation: GPc `CDeepGp` against the prototype and classical MLPs

Run on 2026-11-21 with `gpc.models.DeepGP` built from `main` at ceb4fb5 plus the two fixes in this branch (below). Same seeded data as the JAX prototype and the MLP baselines (`task_src.py` is `task()` from [`../prototype/dgp_train.py`](../prototype/dgp_train.py)).

Settings: hidden width 5, \(M=64\) per layer (16 on the step task), Adam 0.01, 3000 iterations, batch 256, one sample, 3 seeds. Predictions use the façade's `predict` mean path by default (`samples=0`: hidden means propagated, final-layer variance plus noise). Pass `samples=S` for the CIP-0035 mixture. Reproduce with `python gpc_vs_mlp.py`.

## Composite task (10-D input, target a nonlinear function of a 2-D projection; test RMSE / NLL, standardised targets)

| Model | RMSE | NLL |
|---|---|---|
| **GPc deep GP, RBF, 3 layers** | **0.193 ± 0.002** | **−0.344 ± 0.009** |
| GPc deep GP, MLP kernel, 3 layers | 0.226 ± 0.017 | −0.251 ± 0.029 |
| JAX prototype, MLP kernel, 3 layers, inducing points (one seed) | 0.229 | −0.25 |
| GPc deep GP, RBF, 2 layers | 0.249 ± 0.003 | −0.171 ± 0.011 |
| GPc deep GP, MLP kernel, 2 layers | 0.336 ± 0.016 | −0.005 ± 0.036 |
| MLP with the deep GP's block structure, ensemble of 5 | 0.288 | 0.19 |
| MLP with the deep GP's block structure, single network (5 seeds) | 0.340 ± 0.024 | 0.36 |
| Plain ReLU/tanh MLPs, 2–3 hidden × 64, single | 0.70–0.78 | 1.06–1.17 |
| GPc deep GP, 1 layer (RBF / MLP kernel) | 0.892 / 0.915 | 1.30 / 1.33 |
| Exact GP, one layer, MLP kernel | 0.833 | 1.22 |

MLP rows: [`../prototype/mlp_baseline.py`](../prototype/mlp_baseline.py) (tuned on a validation split, retrained on all 2000 points).

## 1-D step task (test RMSE / NLL)

| Model | RMSE | NLL |
|---|---|---|
| **GPc deep GP, MLP kernel, 1 layer** | **0.057 ± 0.005** | **−1.67 ± 0.09** |
| GPc deep GP, RBF, 2 layers | 0.111 ± 0.026 | −0.83 ± 0.25 |
| GPc deep GP, MLP kernel, 3 layers | 0.098 ± 0.026 | −1.18 ± 0.46 |
| GPc deep GP, RBF, 3 layers | 0.143 ± 0.083 | −0.68 ± 0.67 |
| GPc deep GP, MLP kernel, 2 layers | 0.151 ± 0.082 | 2.0 ± 4.9 |
| GPc deep GP, RBF, 1 layer | 0.181 ± 0.013 | −0.26 ± 0.07 |
| Plain ReLU/tanh MLPs, 3 × 64, ensemble of 5 | 0.081–0.086 | −0.95 to −0.96 |
| MLP with the deep GP's block structure, 2–3 blocks, single | 0.084–0.089 | −0.83 to −0.85 |

## Fixes needed to run these (this branch)

1. `CDeepGp::rebuildLayers` built every layer's `CSvgp` on the original `X`, so any hidden width \(D\ne d\) threw "Z and X must have the same number of columns". Layers are now built on `X` pushed through the mean maps.
2. In Cholesky mode, `CSvgp::klDivergence`, `qParamsGrad` and `klQ` re-factorised \(\tilde S=LL^\top\) with `chol()`. With the hidden-layer initialisation (\(L=10^{-2}I\)) this threw "non positive definite" within a few Adam steps at learning rate 0.01 on the composite task. They now use the stored factor \(L\) directly.

`testDeepGp` gains a regression block (\(d=10\), \(D=5\), \(M=64\), Adam 0.01) that fails without either fix.

## Notes

- Default `predict` is the mean path (`samples=0`); CIP-0035's mixture is `predict(..., samples=S)`. Step-task NLLs for deep models may still differ from the prototype when the mean path under-states hidden uncertainty near the jump.
- `jacobian_spread` (V7) now matches the prototype: each probe draws whitened \(\mathbf v\sim\mathcal N(\mathbf 0,I)\) per layer and finite-differences \(\mathbf h\mapsto\mathbf m(\mathbf h)+\mathbf A(\mathbf h)^\top\mathbf v\) (Nyström residual off), independent of the trained \(q(\mathbf v)\).
