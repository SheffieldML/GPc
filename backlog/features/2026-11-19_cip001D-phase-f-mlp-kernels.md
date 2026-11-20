---
id: "2026-11-19_cip001D-phase-f-mlp-kernels"
title: "CIP-001D Phase F: MLP / arc-cosine kernels on GPRegression"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["001D", "0028", "000C"]
owner: "Neil Lawrence"
dependencies:
- "2026-11-18_cip001D-phase-e-notebooks"
tags:
- backlog
- python
- facade
- mlp
- arccos
- kernels
---

# Task: CIP-001D Phase F — MLP / arc-cosine on the façade

> Backlog task for [CIP-001D](../../cip/cip001D.md) Phase F. Unlocks Colab / notebook recipes that compare Williams arcsin (MLP) with RBF on real regression data.

## Description

Phases A–E ship RBF / ExpQuad only. The C++ catalogue already has `CMlpKern` and `CArcCosineKern` (CIP-000C / CIP-0028). Expose them on `GPRegression` (and sparse constructors) via `kernel=` strings so FTC scripts can write `gpc.GPRegression(X, Y, kernel="mlp")`.

Weight-space inducing stays on the sibling task [2026-11-19_cip0028-python-weightspace-notebook](./2026-11-19_cip0028-python-weightspace-notebook.md); this phase owns the kernel catalogue slice and an olympic MLP vs RBF notebook.

## Acceptance Criteria

- [x] `GPRegression(..., kernel="mlp")` builds compound MLP + bias + white; paths include `mlp.weight_variance` / `mlp.bias_variance` / `mlp.variance` (GPy aliases)
- [x] `kernel="arccos"` / `"arccosine"` builds ArcCosine + bias + white; optional `arccos_order=0|1`
- [x] `SparseGPRegression` accepts the same `kernel=` for point inducing
- [x] Unknown kernels still raise `GpcError`; pytest covers construct / path / optimize / predict for MLP
- [x] Notebook `python/examples/mlp_olympic.ipynb`: pods olympic marathon, RBF vs MLP side-by-side via `gpc.plot.plot_1d`
- [x] `cip/cip001D.md` Backlog lists Phase F; `python/README.md` / examples mention the notebook

## Implementation Notes

- Reuse the RBF / ExpQuad branch pattern in `python/src/wrap_gp.cpp`; factor a small compound helper if the ctor grows noisy.
- FTC keeps bias + white for GPy-familiar compounds. Weight-space compounds (no Bias) are the CIP-0028 Python task.
- Do not invent a second naming table — CIP-0014 already maps `weightVariance` ↔ `weight_variance`.

## Related

- CIP: 001D (façade); 0028 (MLP weight-space C++); 000C (kernel catalogue)
- Sibling: `2026-11-19_cip0028-python-weightspace-notebook`

## Progress Updates

### 2026-11-19

Task created (In Progress) on `cip001D-phase-f-mlp-weightspace`; implements with CIP-0028 Python binding in the same branch.
