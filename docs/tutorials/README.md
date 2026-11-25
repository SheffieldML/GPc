# GPc tutorials (native share format)

These pages are the CIP-0015 delivery format: **CLI + gnuplot**, with optional short C++ API demos. GPy notebooks are the **spec/checklist** only — not the runtime.

Build `gp` / `ivm` / `gplvm` / `bgplvm` first (see [README](../../README.md) and [build-and-ci.md](../build-and-ci.md)). Cap optimiser iterations with `-#` while exploring ([optimisers.md](../optimisers.md)).

## Tutorials

| Tutorial | Executable | Data | GPy counterpart |
|---|---|---|---|
| [1D regression (sinc)](./regression_1d.md) | `gp` (+ `demo_regression_1d`) | `examples/sinc.csv` (`.svml` legacy) | `basic_gp` / `toy_rbf_1d` |
| [Hyperparameter posterior samples (sinc)](./hmc_sinc.md) | `gp sample` (+ `gpc.inference.hmc`) | `examples/sinc.csv` | `GPy.inference.mcmc.HMC` ([CIP-003B](../../cip/cip003B.md)) |
| [Sparse 1D (DTCVAR)](./sparse_regression_1d.md) | `gp` (+ `demo_sparse_dtcvar`) | `examples/spgp1d.csv` (`.svml` legacy) | `sparse_gp_regression` |
| [Compound + paths](./compound_paths.md) | `demo_compound_paths` (+ `gp -k`) | `examples/sinc.csv` | `basic_kernels` / `models_basic` |
| [Coregionalization toy](./coregionalization_toy.md) | `demo_coregionalization_toy` | synthesised in-process | `coregionalization_toy` |
| [IVM unit square](./ivm_unitsquare.md) | `ivm` | `examples/unitsquaregp.svml` | Sparse classification (active set) |
| [CGp EP / Laplace classification](./cgp_ep_classification.md) | `testEp` | fixtures in `cip/cip0033/` | Dense EP or Laplace on `CGp` FTC (CIP-0033) |
| [Likelihood zoo](./likelihood_zoo.md) | `testLikelihoodZoo` | SciPy / GH fixtures | Student-t / Poisson / skew… ([CIP-003A](../../cip/cip003A.md)) |
| [IVM ARD clusters](./ivm_ard_clusters.md) | `ivm` | `examples/ard_gaussian_clusters.svml` | ARD / input selection |
| [IVM semi-supervised crescent](./ivm_semisupervised.md) | `ivm` | `examples/semisupercrescent.svml` | ncnm / semi-supervised |
| [GPLVM oil 100](./gplvm_oil100.md) | `gplvm` | `examples/oilTrain100.svml` | `gplvm_oil_100` |

## Bayesian latent models (outside CIP-0015)

Linked here so the tutorial index stays the discovery surface.

| Tutorial | Executable | Data | GPy counterpart |
|---|---|---|---|
| [Bayesian GPLVM oil 100](./bgplvm_oil100.md) | `bgplvm` | `examples/oilTrain100.svml` | `BayesianGPLVM` |
| [SVI Bayesian GPLVM](./bgplvm_svi.md) | `bgplvm --inference svi` | `examples/oilTrain100.svml` | uncollapsed / no-psi ([CIP-0030](../../cip/cip0030.md)) |
| [Deep GP (DSVI)](./deepgp.md) | `deepgp` | `examples/sinc.svml` | deep GP ([CIP-0035](../../cip/cip0035.md)) |
| [MRD two-view smoke](./mrd.md) | `demo_mrd` | synthesised in-process | `MRD` |

## GPy ↔ GPc checklist (CIP-0015)

| GPy tutorial / example | GPc status |
|---|---|
| `basic_gp` / `toy_rbf_1d` / olympics-style regression | Done — [regression_1d](./regression_1d.md) |
| `basic_kernels` (composition) | Done — [compound_paths](./compound_paths.md); CLI `-k matern32` / `stdperiodic` / … |
| `models_basic` (constrain / fix) | Done — [compound_paths](./compound_paths.md) (CIP-0014 paths) |
| `sparse_gp_regression` / `sparse_GP_regression_1D` | Done — [sparse_regression_1d](./sparse_regression_1d.md) (DTCVAR default; FITC not tutorial default) |
| `coregionalized_regression_tutorial` / `coregionalization_toy` / multiple outputs | Done — [coregionalization_toy](./coregionalization_toy.md) + [coregionalize.md](../coregionalize.md) |
| `basic_classification` (EP) | Dense EP — [cgp_ep_classification](./cgp_ep_classification.md); sparse — [ivm_unitsquare](./ivm_unitsquare.md) |
| `gplvm_oil_100` | Done — [gplvm_oil100](./gplvm_oil100.md) |
| `BayesianGPLVM` | Owned by [CIP-0017](../../cip/cip0017.md) — [bgplvm_oil100](./bgplvm_oil100.md) |
| `MRD` | Owned by [CIP-0018](../../cip/cip0018.md) — [mrd](./mrd.md) |
| EP/Laplace in `CGp`, Poisson, Student-t, hetero, warped, mean functions, SVI/HMC, integral kernels, FITC-as-default | **Not in subset** (CIP-0015 non-goals) |
| Python / Jupyter share format | Optional ease-of-use only — [CIP-001D](../../cip/cip001D.md) / [REQ-000E](../../requirements/req000E_gpy-facing-python-facade.md); which GPy tutorials have `gpc` notebooks vs deferred façades: [`python/docs/subset_checklist.md`](../../python/docs/subset_checklist.md) |

Swagger/BVH mocap remains deferred (MATLAB visualisation path).

## Design

- [CIP-0015](../../cip/cip0015.md) · [REQ-0009](../../requirements/req0009_gpy-tutorial-parity-native.md)
- Bayesian GPLVM: [CIP-0017](../../cip/cip0017.md) · [REQ-000B](../../requirements/req000B_bayesian-gplvm-psi.md)
- MRD: [CIP-0018](../../cip/cip0018.md) · [REQ-000C](../../requirements/req000C_manifold-relevance-determination.md)
- Parameter paths: [param-paths.md](../param-paths.md)
- Optimiser budgets: [optimisers.md](../optimisers.md)
- Multi-output layout: [coregionalize.md](../coregionalize.md)
