---
id: "2026-11-22_cip001D-kern-K-and-plot"
title: "Façade Kern.K / thin Kern.plot for basic_kernels parity"
status: "Completed"
priority: "Medium"
created: "2026-11-22"
last_updated: "2026-11-22"
completed: "2026-11-22"
category: "features"
related_cips: ["001D", "0038"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- python
- facade
- kernels
- gpy
- plot
---

# Task: `gpc.kern` — `K(X)` and thin `plot` for `basic_kernels`

> Follow-on after [CIP-0038](../../cip/cip0038.md) catalogue / compounds / `active_dims`. The `basic_kernels` tutorial still cannot call `k.K` / `k.plot` the way GPy’s notebook does.

## Description

`basic_kernels` on the façade now covers the CIP-0038 catalogue, sum/product compounds, `active_dims` on 2-D data, and the CIP-0027 Bias vs signal-only comparison — with an explicit note that `gpc.kern` has no `k.plot` / `k.K` yet.

GPy’s [`basic_kernels`](https://github.com/SheffieldML/notebook/blob/main/GPy/basic_kernels.ipynb) teaches kernels by evaluating and plotting covariance (`k.K(X)`, `k.plot(...)`) before attaching a model. Today `gpc.kern.*` objects are thin `to_parts()` descriptors; covariance maths only run once a `GP*` constructor builds a `CKern`. Close that gap without claiming full GPy plot / paramz identity ([CIP-001D](../../cip/cip001D.md) already rejected a full `m.plot()` stack).

Ship:

1. **`Kern.K(X, X2=None) → ndarray`** — evaluate the covariance via a binding that builds `CKern` from `to_parts()` + a dummy/design `X` (reuse `buildKernFromPart` / compound assembly in `wrap_gp.cpp`), then `compute` into a NumPy array. Honour `active_dims`.
2. **Thin `Kern.plot`** under `gpc[plot]` (or `gpc.plot.plot_kern`) — matplotlib helper over `K`, enough for the GPy notebook cells (1-D slice / small grid). Not a core `import gpc` dependency.
3. **Docs / notebooks** — drop the “no `k.K` / `k.plot` yet” note from `basic_kernels` once green; mention in side-by-side / subset checklist.

## Acceptance Criteria

- [x] `k.K(X)` and `k.K(X, X2)` return `(n, n2)` float arrays for atomic, `Add`, and `Prod` kernels (incl. ARD / `active_dims` smoke)
- [x] Pytest compares a few kernels (e.g. RBF, Matérn32, RBF×Linear) to a NumPy / GPy reference within tolerance
- [x] Thin plot path works with matplotlib (`gpc[plot]`); documented as optional
- [x] `basic_kernels` notebook (package and/or SheffieldML/notebook) uses `k.K` / plot and removes the explicit gap note
- [x] Help/docstrings list `K` (and plot helper) without claiming full GPy plot API parity

## Implementation Notes

- Binding: `gpc._gpc.kern_K(parts, X, X2=None)` in `wrap_gp.cpp` (temporary `CCmpndKern`, no GP).
- `Kern.K` / `Kern.plot` on the façade; `gpc.plot.plot_kern` for the matplotlib slice.
- Assert and Verify: dimension checks at the binding boundary; do not silently reshape `active_dims` mismatches.

## Related

- CIP: 001D, 0038
- Notebooks: [`python/examples/basic_kernels_compound.ipynb`](../../python/examples/basic_kernels_compound.ipynb); SheffieldML/notebook `GPc/basic_kernels.ipynb` (sync later)
- Docs: [`python/docs/subset_checklist.md`](../../python/docs/subset_checklist.md)

## Progress Updates

### 2026-11-22

Task created (Ready): façade gap after CIP-0038 catalogue landing.

Implemented on `cip001D-kern-K-and-plot`: `kern_K` binding, `Kern.K` / thin `plot`, pytest `test_kern_K`, notebook + checklist + side-by-side. Status → Completed.
