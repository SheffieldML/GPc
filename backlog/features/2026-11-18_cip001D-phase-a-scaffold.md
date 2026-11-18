---
id: "2026-11-18_cip001D-phase-a-scaffold"
title: "CIP-001D Phase A: pybind11 package + NumPy/CMatrix bridge"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001D"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- python
- pybind11
- gpy
---

# Task: CIP-001D Phase A scaffold

> Backlog task for [CIP-001D](../../cip/cip001D.md) Phase A.

## Description

Stand up installable/importable `gpc` package with pybind11 extension `_gpc`, link against `libgp`, and ship NumPy ↔ `CMatrix` copy helpers with pytest round-trips.

## Acceptance Criteria

- [x] `python/` layout with `gpc` package and pybind11 sources
- [x] Makefile (or equivalent) builds `_gpc` against existing BLAS/`libgp`
- [x] `Matrix.from_numpy` / `to_numpy` round-trip for 2-D float64
- [x] Loud failure on unsupported array rank
- [x] `make test` (or pytest) green in the worktree

## Implementation Notes

- Prefer makefile-driven build matching the rest of GPc; scikit-build can come later for wheels.
- Copy by default (row-major NumPy ↔ column-major `CMatrix`); no Python in SCG.

## Related

- CIP: 001D
- Requirement: REQ-000E

## Progress Updates

### 2026-11-18

Task created; worktree `cip001D-python-facade` opened; Phase A implementation started.

### 2026-11-18 (done)

`python/` scaffold + NumPy/`CMatrix` bridge + pytest green (`make -C python test`). Status → Completed.
