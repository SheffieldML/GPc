---
id: "2026-11-19_cip0028-python-weightspace-notebook"
title: "CIP-0028: Python gpc weight-space API + illustrative notebook"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028", "001D"]
owner: "Neil Lawrence"
dependencies:
- "2026-11-19_cip0028-phase-3-docs-stability"
- "2026-11-18_cip001D-phase-c-sparse"
tags:
- backlog
- python
- jupyter
- sparse
- inducing
- weight-space
- cip0028
---

# Task: CIP-0028 — `gpc` weight-space binding and notebook

## Description

C++ CIP-0028 Phases 0–3 land on the CIP-0028 PR. The Python façade still only exposes RBF + **point** inducing via `SparseGPRegression(X, Y, Z)`. There is no `gpc` notebook that shows what ridges do.

Extend the binding enough to drive DTCVAR with weight-space inducing from Python, then ship one illustrative notebook (toy 1-D) that compares points vs ridge weight-space: bound / objective, free centres, and a predictive plot.

The NumPy prototype under `cip/cip0028/prototype/deepws/` stays evidence-only; the notebook must go through `gpc` → C++.

## Acceptance Criteria

- [ ] `SparseGPRegression` (or a thin sibling) accepts `kernel="mlp"` / `"arccos"` and `inducing="weightspace"` (default remains RBF + `points`)
- [ ] Optional: `window_width`, `inducing_fixed`, arccos `order`; centres owned by inducing object (Z may be a size hint / unused for weight-space)
- [ ] Compound rule matches C++: one signal + optional white; noise via β / likelihood variance
- [ ] Pytest smoke: construct, `objective` finite, `optimize`, `predict` for MLP + weightspace DTCVAR
- [ ] Notebook under `python/examples/` (e.g. `weightspace_ridge_1d.ipynb`) that illustrates ridges vs points for a reader who has not read the CIP
- [ ] `python/README.md` / examples index mention the notebook; plotting via existing `gpc.plot.plot_1d` extras

## Implementation Notes

- Today `wrap_gp.cpp` always calls `addDefaultCompoundKernel` (SE + bias + white) and never `setInducingType`.
- Prefer extending `SparseGPRegression` kwargs over a second class unless the ctor becomes unreadable.
- For weight-space, `Z` rows can set `numActive`; ridge init lives in `CInducingWeightSpace::makeRidgeInit`.
- Keep pods / matplotlib optional (CIP-001D Phase E pattern).
- Depends on CIP-0028 C++ merge (`cip0028-weightspace-inducing`); can land this backlog file on main first.

## Related

- CIP: 0028 (core PR); 001D (Python façade)
- Depends: CIP-0028 Phase 3 on merge; CIP-001D sparse Phase C
- Prototype (not the deliverable): `cip/cip0028/prototype/` (after CIP-0028 merges)

## Progress Updates

### 2026-11-19
Task created (Proposed) for a follow-on after the CIP-0028 C++ PR; tracked on main separately from that PR.
