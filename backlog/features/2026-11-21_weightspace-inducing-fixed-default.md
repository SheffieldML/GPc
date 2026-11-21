---
id: "2026-11-21_weightspace-inducing-fixed-default"
title: "Weight-space / sparse façade: inducing_fixed default fights CIP-0028 mean"
status: "In Progress"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0028", "001D"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- python
- weightspace
- inducing
- api
---

# Task: Fix inducing_fixed default for weight-space (and demos)

## Description

C++ `CGp` defaults `inducingFixed = false`. The Python `SparseGPRegression` façade defaults `inducing_fixed=True`, which freezes point locations and weight-space ridge centres. For CIP-0028 weight-space ridges the finite-\(M\) posterior mean is a bank of smoothed erf units — that geometry cannot form if centres stay fixed. Notebooks (`weightspace_ridge_1d`) then look underfit after a short optimise.

Follow-on after closed CIP-0028 / CIP-001D (not a reopen): façade default policy + demo/docs.

## Acceptance Criteria

- [ ] `inducing="weightspace"` defaults to free centres (`inducing_fixed=False`), or an equally clear opt-in that demos use
- [ ] `weightspace_ridge_1d.ipynb` and `python/README.md` document free vs fixed centres and fixed \(s\)
- [ ] Point-inducing callers that relied on fixed \(Z\) remain documented (`inducing_fixed=True` explicit)

## Implementation Notes

Prefer: weight-space construction path sets free centres; keep `inducing_fixed=True` as the default only for `inducing="points"` if changing the global default is too breaking. Tests that assert `inducing_fixed is True` for point sparse need updating if the global default flips.

## Related

- CIP: 0028, 001D
- Sibling: `2026-11-21_weightspace-learn-window-width`

## Progress Updates

### 2026-11-21

Task created after reviewing `weightspace_ridge_1d` underfit (centres frozen).
