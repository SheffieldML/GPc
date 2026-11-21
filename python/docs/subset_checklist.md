# CIP-0015 tutorial checklist → `gpc` (CIP-001D Phase D)

Native demos/CLI remain the primary share format ([CIP-0015](../../cip/cip0015.md)). This table maps the same GPy checklist items to the Python façade where it exists.

## In the Python subset (notebooks / recipes)

| GPy tutorial / example | Native GPc | `gpc` Python |
|---|---|---|
| `basic_gp` / `toy_rbf_1d` | [regression_1d](../../docs/tutorials/regression_1d.md) | `gpc.models.GPRegression(X, Y, kernel=k)` with `gpc.kern.*` — [side-by-side](./gpy_side_by_side.md); notebook [`examples/toy_1d_regression.ipynb`](../examples/toy_1d_regression.ipynb) teaches `predict` / `predict_noiseless` (CIP-0031) |
| `sparse_GP_regression_1D` | [sparse_regression_1d](../../docs/tutorials/sparse_regression_1d.md) (DTCVAR) | Notebook [`examples/sparse_gp_regression_1d.ipynb`](../examples/sparse_gp_regression_1d.ipynb); `approx="dtcvar"` / `"fitc"` / `"dtc"` |
| Olympics-style 1-D | CLI / pods data via scripts | Notebooks: `examples/olympic_marathon.ipynb`, `examples/mlp_olympic.ipynb` (need `pods`); `plot_1d(..., noiseless=)` |
| `basic_kernels` / custom compounds | CIP-000C catalogue + paths | Notebook [`examples/basic_kernels_compound.ipynb`](../examples/basic_kernels_compound.ipynb); `gpc.kern.RBF/ExpQuad/White/Bias/MLP/ArcCosine` + `+` |
| GPy `ExpQuad` naming | [param-paths](../../docs/param-paths.md) | Notebook [`examples/expquad_regression.ipynb`](../examples/expquad_regression.ipynb); script twin `examples/expquad_regression.py` |
| MLP / arcsin vs RBF | CIP-000C / CIP-001D Phase F | [`examples/mlp_olympic.ipynb`](../examples/mlp_olympic.ipynb) (year-scale `weight_variance` init in notebook; GPy defaults stay `1`) |
| Weight-space inducing | [CIP-0028](../../cip/cip0028.md) | [`examples/weightspace_ridge_1d.ipynb`](../examples/weightspace_ridge_1d.ipynb) |

Install: see [`../README.md`](../README.md).

## Public tutorial notebooks

Teaching / Colab ports of this checklist live in the SheffieldML notebook repo:

- [SheffieldML/notebook `GPc/`](https://github.com/SheffieldML/notebook/tree/main/GPc) ([CIP-0007](https://github.com/SheffieldML/notebook/blob/main/cip/cip0007.md))
- Package-local smoke notebooks remain under [`../examples/`](../examples/)

## Not yet in the Python subset (deferred — use native tutorials)

These GPy-online / CIP-0015 checklist items are **intentionally skipped** as `gpc` notebooks for now. Native CLI/demos cover them; Python wraps are follow-ons owned by the CIPs below (not close blockers for [CIP-001D](../../cip/cip001D.md) / [REQ-000E](../../requirements/req000E_gpy-facing-python-facade.md)).

| GPy tutorial / example | Native GPc today | Owning CIP / backlog (Python later) | Notes |
|---|---|---|---|
| Coregionalization / `build_XY` / ICM | [coregionalization_toy](../../docs/tutorials/coregionalization_toy.md), [coregionalize.md](../../docs/coregionalize.md) | [CIP-0015](../../cip/cip0015.md) Phase A helpers exist natively; façade wrap listed under CIP-001D follow-ups | Multi-output index column / `stampOutputIndex` not bound |
| `basic_classification` (EP) | IVM substitute — [ivm_unitsquare](../../docs/tutorials/ivm_unitsquare.md) | CIP-001D deferred (“Wrapping IVM…”) | No EP/Laplace in `CGp`; no `gpc` classifier |
| GPLVM | [gplvm_oil100](../../docs/tutorials/gplvm_oil100.md) | CIP-001D follow-ups | MAP GPLVM CLI only |
| Bayesian GPLVM (collapsed / psi) | [bgplvm_oil100](../../docs/tutorials/bgplvm_oil100.md) | [CIP-0017](../../cip/cip0017.md) (Closed native); Python still follow-on | |
| SVI Bayesian GPLVM | [bgplvm_svi](../../docs/tutorials/bgplvm_svi.md) | [CIP-0030](../../cip/cip0030.md); backlog [`2026-11-20_cip0030-phase4b-python-facade`](../../backlog/features/2026-11-20_cip0030-phase4b-python-facade.md) (Ready) | CLI-first today |
| MRD | [mrd](../../docs/tutorials/mrd.md) | [CIP-0018](../../cip/cip0018.md) (Closed native); CIP-001D follow-ups | |
| SVGP / sparse variational | [svgp](../../docs/tutorials/svgp.md) | [CIP-002A](../../cip/cip002A.md); Phase 5 surfaces backlog [`2026-11-19_cip002A-phase5-surfaces`](../../backlog/features/2026-11-19_cip002A-phase5-surfaces.md) | Tutorial native; Python façade deferred |
| `models_basic` constrain / `fix` | CIP-0014 paths | Backlog [`2026-11-21_cip001D-python-fix-by-path`](../../backlog/features/2026-11-21_cip001D-python-fix-by-path.md) | Path get/set only today |
| Sparse `likelihood.variance` teaching | CIP-0027 | Backlog [`2026-11-21_cip0027-python-sparse-likelihood-variance`](../../backlog/features/2026-11-21_cip0027-python-sparse-likelihood-variance.md) | Docs/façade teaching follow-on |
| Poisson / Student-t / hetero / warped / HMC / mean functions | — | CIP-0015 / REQ-000E **non-goals** | Out of scope for the first subset |
| Full GPy/`paramz` object graph, regex indexing | — | REQ-000E non-goals | Ergonomics ≈ recipes, not API identity |

**Share-format reminder:** Jupyter is an optional ease-of-use surface. GPy notebooks remain the **spec/checklist**; CIP-0015 native CLI + gnuplot remain the primary verified share format ([docs/tutorials/README.md](../../docs/tutorials/README.md)).
