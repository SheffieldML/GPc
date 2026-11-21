---
id: "2026-11-21_cip001D-python-fix-by-path"
title: "CIP-001D follow-on: wrap fix/unfix by path on the Python façade"
status: "Ready"
priority: "Medium"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["001D", "0014"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- python
- fix
- constraints
- gpy
---

# Task: Python façade `fix` / `unfix` by path

> Gap in closed [CIP-001D](../../cip/cip001D.md): C++ already has [CIP-0014](../../cip/cip0014.md) `CGp::fix` / `unfix`; the façade only wraps path get/set. Checklist: `python/docs/subset_checklist.md` (`models_basic` constrain/fix).

## Description

Expose GPy-familiar parameter fixing on `GPRegression` / `SparseGPRegression` (and any thin wrappers that already forward `set_by_path`):

- `fix(path)` / `unfix(path)` bound to `CGp::fix` / `unfix`
- Optional: read-only “is fixed” via existing `ModelDescription` / `ParamRow.fixed` if already available in describe

Do **not** clone paramz `constrain_*` beyond what CIP-0014 already provides in C++ unless a path already exists.

## Acceptance Criteria

- [ ] `m.fix("rbf.variance")` (or path-equivalent) drops the parameter from the next `optimize` vector; value retained
- [ ] `m.unfix(...)` restores it
- [ ] Pytest covers FTC and at least one sparse path; negative path throws loudly
- [ ] `python/docs/subset_checklist.md` and side-by-side note that fix is wrapped
- [ ] Docstrings / `help()` mention `fix` / `unfix` ([CIP-002F](../../cip/cip002F.md) style)

## Implementation Notes

- Bind in `python/src/wrap_gp.cpp`; thin methods on `gpc.models` if display layer needs them.
- Prefer canonical paths (`likelihood.variance`, `rbf.lengthscale`); accept `kern.` prefix if C++ already does.
- Assert and Verify: compare optimiser param count before/after fix against a native `CGp` reference or known length.

## Related

- CIP: 001D, 0014
- Checklist: [`python/docs/subset_checklist.md`](../../python/docs/subset_checklist.md)

## Progress Updates

### 2026-11-21

Task created Ready after Python-compat gap review (finish 001D wrapping; not a new CIP).
