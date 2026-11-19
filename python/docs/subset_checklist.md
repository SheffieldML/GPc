# CIP-0015 tutorial checklist → `gpc` (CIP-001D Phase D)

Native demos/CLI remain the primary share format ([CIP-0015](../../cip/cip0015.md)). This table maps the same GPy checklist items to the Python façade where it exists.

| GPy tutorial / example | Native GPc | `gpc` Python |
|---|---|---|
| `basic_gp` / `toy_rbf_1d` | [regression_1d](../../docs/tutorials/regression_1d.md) | `gpc.GPRegression(X, Y)` — [side-by-side](./gpy_side_by_side.md) |
| `sparse_gp_regression` | [sparse_regression_1d](../../docs/tutorials/sparse_regression_1d.md) (DTCVAR) | `gpc.SparseGPRegression(X, Y, Z)` — default `approx="dtcvar"`; also `"fitc"` / `"dtc"` |
| Olympics-style 1-D | CLI / pods data via scripts | Notebook: `examples/olympic_marathon.ipynb` (Phase E; needs `pods`) |
| `basic_kernels` / custom compounds | CIP-000C catalogue + paths | Not yet — default RBF+bias+white only |
| `models_basic` constrain/fix | CIP-0014 `fix` / paths | Path get/set only; fix masks not wrapped yet |
| Coregionalization / `build_XY` | CIP-0015 Phase A (native) | Not in Python subset yet |
| `basic_classification` (EP) | IVM substitute | Not in Python subset |
| GPLVM / Bayesian GPLVM / MRD | Native tutorials | Not in Python subset (follow-ups) |
| Poisson / hetero / SVI / HMC | Out of scope | Out of scope |

Install: see [`../README.md`](../README.md).
