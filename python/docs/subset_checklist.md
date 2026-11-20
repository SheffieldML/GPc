# CIP-0015 tutorial checklist → `gpc` (CIP-001D Phase D)

Native demos/CLI remain the primary share format ([CIP-0015](../../cip/cip0015.md)). This table maps the same GPy checklist items to the Python façade where it exists.

| GPy tutorial / example | Native GPc | `gpc` Python |
|---|---|---|
| `basic_gp` / `toy_rbf_1d` | [regression_1d](../../docs/tutorials/regression_1d.md) | `gpc.models.GPRegression(X, Y, kernel=k)` with `gpc.kern.*` — [side-by-side](./gpy_side_by_side.md); string `gpc.GPRegression(X, Y)` still OK |
| `sparse_gp_regression` | [sparse_regression_1d](../../docs/tutorials/sparse_regression_1d.md) (DTCVAR) | `gpc.models.SparseGPRegression(X, Y, Z, kernel=k)` — default `approx="dtcvar"`; also `"fitc"` / `"dtc"` |
| Olympics-style 1-D | CLI / pods data via scripts | Notebook: `examples/olympic_marathon.ipynb` (Phase E; needs `pods`) |
| `basic_kernels` / custom compounds | CIP-000C catalogue + paths | `gpc.kern.RBF/ExpQuad/White/Bias/MLP/ArcCosine` + `+` (first subset) |
| `models_basic` constrain/fix | CIP-0014 `fix` / paths | Path get/set only; fix masks not wrapped yet |
| Coregionalization / `build_XY` | CIP-0015 Phase A (native) | Not in Python subset yet |
| `basic_classification` (EP) | IVM substitute | Not in Python subset |
| GPLVM / Bayesian GPLVM / MRD | Native tutorials | Not in Python subset (follow-ups) |
| Poisson / hetero / SVI / HMC | Out of scope | Out of scope |

Install: see [`../README.md`](../README.md).
