# CIP-0035 prototype: deep GPs with point and weight-space inducing (JAX)

Evidence for [CIP-0035](../../cip0035.md). JAX, float64, Adam. Not production code: it is the numerical oracle and the source of the numbers quoted in the CIP.

Requires `jax`, `optax`, `numpy`, `scipy`. `check_dgp.py` also imports `ridge.py` / `wsfeatures.py` from [`../../cip0028/prototype/deepws/`](../../cip0028/prototype/deepws/).

| File | What it does | Output |
|---|---|---|
| `dgp.py` | Deep GP model. Layer kinds: `ip` (inducing points), `ridge` (fixed ridge windows), `ridge_learn` (learned ridge windows), `hermite` (fixed Hermite windows), `rf` (random erf units, finite Bayesian net). Identity / PCA mean functions, per-layer variance \(\chi/(L-1)\), Cholesky-parameterised \(q(\mathbf v)\), local reparameterisation or one weight draw per minibatch, optional residual \(k-\|\phi\|^2\). | — |
| `check_dgp.py` | Window features vs NumPy reference; \(L=1\) ELBO \(\le\) exact log marginal for each kind | stdout |
| `dgp_jacobian.py` | Prior end-to-end Jacobian spread vs depth | `dgp_jacobian.json` |
| `dgp_train.py` | Units per layer × depth × kind, composite (10-D) and 1-D step tasks | `dgp_train_*.json` |
| `dgp_diag.py` | Fixed windows with and without the residual term; variance captured per layer | `dgp_diag.json` |
| `dgp_learn.py` | Learned ridge windows (K_uu refactorised per step) | `dgp_learn.json` |
| `dgp_wide.py` | Hidden widths 16 and 32 | `dgp_wide.json` |
| `dgp_timing.py` | ms per Adam step vs a deep MLP | table below |
| `mlp_baseline.py` | Classical MLPs (tuned on validation, retrained on all data, 5 seeds, ensembles) | `mlp_baseline_composite.log`, `mlp_baseline_step.json` |

Run from this directory, e.g. `python dgp_train.py composite`.

## Timing (2-core sandbox, CPU, \(L=5\), width 8, batch 256; ms per Adam step)

| \(M\) | deep MLP | windows, one weight draw, no residual | + residual | per-point sampling | inducing points |
|---|---|---|---|---|---|
| 64 | 2.8 | 2.4 | 3.4 | 4.2 | 9.4 |
| 256 | 7.2 | 10.9 | 14.8 | 19.8 | 62 |
| 1024 | 22 | 103 | 161 | 211 | 1063 |

## Caveats

One synthetic composite task (10-D input, target a nonlinear function of a 2-D projection, \(N=2000\)/1000, one seed for the deep GPs) and a 1-D step task (two seeds). Deep GPs were not tuned (Adam 0.01, 2500–3000 steps); MLPs were tuned on a validation split. The prototype has no window-width clamp; GPc already bounds \(s\in[0.05,1]\).
