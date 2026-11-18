---
id: "2026-11-18_cip001D-phase-b-gp"
title: "CIP-001D Phase B: FTC GPRegression + path get/set"
status: "In Progress"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip001D-phase-a-scaffold"]
tags:
- backlog
- python
- gp
- gpy
---

# Task: CIP-001D Phase B FTC GPRegression

> Backlog task for [CIP-001D](../../cip/cip001D.md) Phase B.

## Description

Expose FTC Gaussian `GPRegression` with default RBF+bias+white compound kernel, CIP-0014 path get/set, `optimize`, `predict`, and pytest vs finite-objective smoke behaviour. Side-by-side GPy doc fragment follows once API stabilises.

## Acceptance Criteria

- [x] `gpc.GPRegression(X, Y)` constructs FTC + Gaussian noise
- [x] `set_by_path` / `get_by_path` / `param_paths` work for `rbf.*` paths
- [x] `optimize` and `predict` return finite NumPy results
- [x] Shape mismatches raise `GpcError`
- [x] Pytest coverage for the above
- [ ] Side-by-side GPy doc fragment (remaining)
- [ ] Optional: pytest vs native C++ objective on fixed seed (remaining)

## Implementation Notes

Holder owns `CMatrix` X/Y, `CCmpndKern`, `CGaussianNoise`, and `CGp`; no Python in SCG.

## Related

- CIP: 001D

## Progress Updates

### 2026-11-18

Task created; initial `GPRegression` wrapper and tests landing on `cip001D-python-facade`.
