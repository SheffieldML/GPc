---
id: "2026-11-20_cip001D-gpy-form-kern-models"
title: "CIP-001D: gpc.kern / models / compound + API shape"
status: "Completed"
priority: "High"
created: "2026-11-20"
completed: "2026-11-20"
owner: "Neil Lawrence"
related_cips: ["001D"]
dependencies: ["2026-11-18_cip001D-phase-b-gp"]
---

# Task: CIP-001D GPy-form kern / models API

> Backlog task for [CIP-001D](../../cip/cip001D.md) API sketch completion.

## Description

Ship the CIP-001D sketched surface: `gpc.kern.*` factories, compound `+`, `gpc.models.GPRegression(..., kernel=k)`, and keep string-kernel defaults for signal+bias+white. Update side-by-side docs and subset checklist. Interactive `help()` / rich docstrings are **out of scope** (follow-on documentation CIP).

## Acceptance Criteria

- [x] `gpc.kern.RBF/ExpQuad/White/Bias/MLP/ArcCosine` with `__add__` → parts list
- [x] `gpc.models.GPRegression` / `SparseGPRegression` accept kern objects
- [x] String `kernel=` path unchanged (default compound)
- [x] Pytest for compound recipes; existing suite still green
- [x] Side-by-side + subset checklist updated

## Related

- CIP: 001D

## Progress Updates

### 2026-11-20

Implemented on `cip001D-python-facade` after fast-forward from `main`.
