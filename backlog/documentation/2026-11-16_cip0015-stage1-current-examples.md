---
id: "2026-11-16_cip0015-stage1-current-examples"
title: "Stage 1: Redo current README examples as native tutorials/demos (CIP-0015)"
status: "Ready"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "documentation"
related_cips: ["0015"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- tutorials
- demos
- gpy
- documentation
---

# Task: CIP-0015 Stage 1 — current examples as tutorials

## Description

First implementation cut for [CIP-0015](../../cip/cip0015.md): re-express the **existing** README / `examples/*.svml` workflows as the native share format (`docs/tutorials/` + optional thin C++ API demos). This does **not** require Phase A (`build_XY`); it proves markdown + CLI/gnuplot (and path-aware demos where useful) before multi-output glue.

### In scope (current examples)

| Existing workflow | Tutorial / demo target | GPy checklist note |
|---|---|---|
| `./gp learn` + gnuplot on `sinc.svml` | `docs/tutorials/regression_1d.md`; optional `examples/demo_regression_1d.cpp` | ≈ `basic_gp` / `toy_rbf_1d` |
| `./gp learn` sparse on `spgp1d.svml` (DTCVAR) | `docs/tutorials/sparse_regression_1d.md`; optional `demo_sparse_dtcvar.cpp` | ≈ `sparse_gp_regression` |
| `./ivm` `unitsquaregp` | `docs/tutorials/ivm_unitsquare.md` | classification substitute for EP tutorials |
| `./ivm` `ard_gaussian_clusters` | `docs/tutorials/ivm_ard_clusters.md` | ARD / input selection story |
| `./ivm` `semisupercrescent` (+ labelled) | `docs/tutorials/ivm_semisupervised.md` | ncnm / semi-supervised |
| `./gplvm` `oilTrain100` | `docs/tutorials/gplvm_oil100.md` | ≈ `gplvm_oil_100` |

Also: a `docs/tutorials/README.md` index that starts the GPy↔GPc checklist (Stage 1 rows filled; others “planned” or “out of scope”).

### Out of scope for Stage 1

- Swagger/BVH mocap (MATLAB helper path) — defer
- Coregionalization toy / `build_XY` — Phase A + later Phase B
- New kernel CLI polish — Phase D
- Changing numeric behaviour of `gp`/`ivm`/`gplvm` beyond demo wiring

## Acceptance Criteria

- [ ] `docs/tutorials/` exists with one page per in-scope workflow above
- [ ] Each page: build prerequisites, exact commands (capped iterations ok), gnuplot/plot steps, link to GPy counterpart or “IVM substitute”
- [ ] `docs/tutorials/README.md` indexes pages and starts the GPy checklist table
- [ ] At least sinc and spgp1d have either a documented CLI-only path **or** a compiled `examples/demo_*.cpp` makefile target using `setByPath` where parameters are set
- [ ] README.md root tutorials either point at `docs/tutorials/` or stay in sync (no contradictory commands)
- [ ] No FITC as the sparse tutorial default (DTCVAR / existing working approx only)

## Implementation Notes

- Prefer capped `-#` iterations in tutorial commands (same spirit as current README).
- API demos should be short and fail loudly on shape errors; do not duplicate full CLI option parsing.
- Gnuplot styling already follows GPy Tango in `CClctrl` — reuse.

## Related

- CIP: 0015
- REQ: 0009
- Data: `examples/*.svml`
- Follow-on: [Phase A helpers](../features/2026-11-16_cip0015-phase-a-helpers.md)

## Progress Updates

### 2026-11-16

Task created (Ready) after CIP-0015 Accepted; Stage 1 launch before Phase A.
