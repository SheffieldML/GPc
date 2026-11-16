# GPc tutorials (native share format)

These pages are the CIP-0015 delivery format: **CLI + gnuplot**, with optional short C++ API demos. GPy notebooks are the checklist only — not the runtime.

Build `gp` / `ivm` / `gplvm` first (see [README](../../README.md) and [build-and-ci.md](../build-and-ci.md)). Cap optimiser iterations with `-#` while exploring.

## Current-example tutorials (Stage 1)

| Tutorial | Executable | Data | GPy counterpart |
|---|---|---|---|
| [1D regression (sinc)](./regression_1d.md) | `gp` (+ `demo_regression_1d`) | `examples/sinc.svml` | `basic_gp` / `toy_rbf_1d` |
| [Sparse 1D (DTCVAR)](./sparse_regression_1d.md) | `gp` (+ `demo_sparse_dtcvar`) | `examples/spgp1d.svml` | `sparse_gp_regression` |
| [IVM unit square](./ivm_unitsquare.md) | `ivm` | `examples/unitsquaregp.svml` | EP classification → **IVM substitute** |
| [IVM ARD clusters](./ivm_ard_clusters.md) | `ivm` | `examples/ard_gaussian_clusters.svml` | ARD / input selection |
| [IVM semi-supervised crescent](./ivm_semisupervised.md) | `ivm` | `examples/semisupercrescent.svml` | ncnm / semi-supervised |
| [GPLVM oil 100](./gplvm_oil100.md) | `gplvm` | `examples/oilTrain100.svml` | `gplvm_oil_100` |

## GPy ↔ GPc checklist (partial)

| GPy tutorial / example | Status in GPc |
|---|---|
| `basic_gp` / `toy_rbf_1d` | Stage 1 — [regression_1d](./regression_1d.md) |
| `basic_kernels` | Planned (compound + CIP-000C types) |
| `models_basic` (constrain/fix) | Partial — CIP-0014 paths; demos use `setByPath` |
| `sparse_gp_regression` | Stage 1 — [sparse_regression_1d](./sparse_regression_1d.md) (DTCVAR; not FITC) |
| `basic_classification` (EP) | IVM substitute — [ivm_unitsquare](./ivm_unitsquare.md) |
| Coregionalization tutorials | Phase A helpers + later demo |
| `gplvm_oil_100` | Stage 1 — [gplvm_oil100](./gplvm_oil100.md) |
| Poisson / hetero / SVI / HMC / BGPLVM | Out of scope (CIP-0015 non-goals) |

Swagger/BVH mocap remains deferred (MATLAB visualisation path).

## Design

- [CIP-0015](../../cip/cip0015.md) · [REQ-0009](../../requirements/req0009_gpy-tutorial-parity-native.md)
- Parameter paths: [param-paths.md](../param-paths.md)
- Multi-output layout (later): [coregionalize.md](../coregionalize.md)
