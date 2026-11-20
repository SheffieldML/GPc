---
id: "2026-11-21_cip0015-phase-b-demos"
title: "Phase B: coreg + compound path demos (CIP-0015)"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0015", "000D", "0014"]
owner: "Neil Lawrence"
dependencies: ["2026-11-16_cip0015-phase-a-helpers"]
tags:
- backlog
- demos
- multioutput
- tutorials
- gpy
- features
---

# Task: CIP-0015 Phase B — remaining demos + makefile

## Description

Implement Phase B of [CIP-0015](../../cip/cip0015.md): `demo_coregionalization_toy` and `demo_compound_paths`, makefile targets, and markdown tutorials that map to the GPy checklist entries still open after Stage 1 / Phase A.

## Acceptance Criteria

- [x] `demo_coregionalization_toy` uses `CCoregionalizedRegression` / slices / `stampOutputIndex` on synthesised two-output sinusoids
- [x] `demo_compound_paths` exercises `CCmpndKern` `paramPaths` / `setByPath` / `constrainPositive`
- [x] Makefile `demos` target builds both; `clean` removes binaries
- [x] Tutorials under `docs/tutorials/` + index / checklist update
- [x] Smoke run of both demos succeeds

## Implementation Notes

- Prefer short SCG budgets (~25 iters) so local/CI smoke stays cheap.
- No multi-output CLI in this phase (API demo only for coreg).

## Related

- CIP: 0015
- Also: REQ-0006 / REQ-0009 end-to-end path

## Progress Updates

### 2026-11-21

Task created and delivered with demos + tutorials. Status → Completed.
