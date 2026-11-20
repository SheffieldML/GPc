# GPc tutorials (native share format)

These pages are the CIP-0015 delivery format: **CLI + gnuplot**, with optional short C++ API demos. GPy notebooks are the checklist only — not the runtime.

Build `gp` / `ivm` / `gplvm` / `bgplvm` first (see [README](../../README.md) and [build-and-ci.md](../build-and-ci.md)). Cap optimiser iterations with `-#` while exploring ([optimisers.md](../optimisers.md)).

## Current-example tutorials (Stage 1)

| Tutorial | Executable | Data | GPy counterpart |
|---|---|---|---|
| [1D regression (sinc)](./regression_1d.md) | `gp` (+ `demo_regression_1d`) | `examples/sinc.csv` (`.svml` legacy) | `basic_gp` / `toy_rbf_1d` |
| [Sparse 1D (DTCVAR)](./sparse_regression_1d.md) | `gp` (+ `demo_sparse_dtcvar`) | `examples/spgp1d.csv` (`.svml` legacy) | `sparse_gp_regression` |
| [Compound + paths](./compound_paths.md) | `demo_compound_paths` (+ `gp -k`) | `examples/sinc.csv` | `basic_kernels` / `models_basic` |
| [Coregionalization toy](./coregionalization_toy.md) | `demo_coregionalization_toy` | synthesised in-process | `coregionalization_toy` |
| [IVM unit square](./ivm_unitsquare.md) | `ivm` | `examples/unitsquaregp.svml` | EP classification → **IVM substitute** |
| [IVM ARD clusters](./ivm_ard_clusters.md) | `ivm` | `examples/ard_gaussian_clusters.svml` | ARD / input selection |
| [IVM semi-supervised crescent](./ivm_semisupervised.md) | `ivm` | `examples/semisupercrescent.svml` | ncnm / semi-supervised |
| [GPLVM oil 100](./gplvm_oil100.md) | `gplvm` | `examples/oilTrain100.svml` | `gplvm_oil_100` |

## Bayesian latent models (CIP-0017 / CIP-0018)

Owned outside CIP-0015 Stage 1, but linked here so the tutorial index stays the discovery surface.

| Tutorial | Executable | Data | GPy counterpart |
|---|---|---|---|
| [Bayesian GPLVM oil 100](./bgplvm_oil100.md) | `bgplvm` | `examples/oilTrain100.svml` | `BayesianGPLVM` |
| [SVI Bayesian GPLVM](./bgplvm_svi.md) | `bgplvm --inference svi` | `examples/oilTrain100.svml` | uncollapsed / no-psi path ([CIP-0030](../../cip/cip0030.md)) |
| [MRD two-view smoke](./mrd.md) | `demo_mrd` | synthesised in-process | `MRD` |

## GPy ↔ GPc checklist (partial)

| GPy tutorial / example | Status in GPc |
|---|---|
| `basic_gp` / `toy_rbf_1d` | Stage 1 — [regression_1d](./regression_1d.md) |
| `basic_kernels` | Phase B — [compound_paths](./compound_paths.md) |
| `models_basic` (constrain/fix) | Phase B — [compound_paths](./compound_paths.md) (CIP-0014 paths) |
| `sparse_gp_regression` | Stage 1 — [sparse_regression_1d](./sparse_regression_1d.md) (DTCVAR default; FITC/DTC available) |
| `basic_classification` (EP) | IVM substitute — [ivm_unitsquare](./ivm_unitsquare.md) |
| Coregionalization tutorials | Phase B — [coregionalization_toy](./coregionalization_toy.md) |
| `gplvm_oil_100` | Stage 1 — [gplvm_oil100](./gplvm_oil100.md) |
| `BayesianGPLVM` (oil-style) | [bgplvm_oil100](./bgplvm_oil100.md) ([CIP-0017](../../cip/cip0017.md)) |
| `MRD` | [mrd](./mrd.md) ([CIP-0018](../../cip/cip0018.md)) |
| Poisson / hetero / SVI / HMC | Out of scope (CIP-0015 non-goals) |

Swagger/BVH mocap remains deferred (MATLAB visualisation path).

## Design

- [CIP-0015](../../cip/cip0015.md) · [REQ-0009](../../requirements/req0009_gpy-tutorial-parity-native.md)
- Bayesian GPLVM: [CIP-0017](../../cip/cip0017.md) · [REQ-000B](../../requirements/req000B_bayesian-gplvm-psi.md)
- MRD: [CIP-0018](../../cip/cip0018.md) · [REQ-000C](../../requirements/req000C_manifold-relevance-determination.md)
- Parameter paths: [param-paths.md](../param-paths.md)
- Optimiser budgets: [optimisers.md](../optimisers.md)
- Multi-output layout: [coregionalize.md](../coregionalize.md)
