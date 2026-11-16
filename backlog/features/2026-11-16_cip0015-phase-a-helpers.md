---
id: "2026-11-16_cip0015-phase-a-helpers"
title: "Phase A: build_XY, slices, coregionalized CGp factory (CIP-0015)"
status: "Ready"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "features"
related_cips: ["0015", "000D"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- multioutput
- helpers
- gpy
- features
---

# Task: CIP-0015 Phase A — multi-output helpers and factory

## Description

Implement Phase A of [CIP-0015](../../cip/cip0015.md): GPy-aligned `build_XY` and slice helpers, plus a thin factory that wires ICM/LCM + `indexwhite` into a trainable `CGp` (spherical Gaussian / stacked design per [CIP-000D](../../cip/cip000D.md) and [docs/coregionalize.md](../../docs/coregionalize.md)).

Does **not** block [Stage 1](../documentation/2026-11-16_cip0015-stage1-current-examples.md). Required before the coregionalization toy demo.

## Acceptance Criteria

- [ ] `build_XY` (or equivalent) stacks per-output `X`/`y` and appends integer-valued output index as last column
- [ ] Slice helpers comparable to GPy `get_slices` / `index_to_slices` for per-output row ranges
- [ ] Factory returns a documented ownership model: ICM or LCM + `CIndexWhiteKern` + `CGp` + Gaussian noise on stacked targets
- [ ] Unit tests cover shapes, index cast failures, and basic factory smoke (extend `testCoregionalize` or sibling)
- [ ] Docs updated in `docs/coregionalize.md` (or tutorial cross-link) for the helper API

## Implementation Notes

- Prefer placing helpers next to `createIcmKern` / `createLcmKern` in `CKern.h` / `CKern.cpp` unless linkage clarity needs a dedicated header.
- Index cast rules stay CIP-000D (`llround` + `MATCHTOL`).
- No new approximation type; no Python bindings.

## Related

- CIP: 0015
- Also: 000D / REQ-0006 end-to-end path
- Blocked demos: `demo_coregionalization_toy` (Phase B)

## Progress Updates

### 2026-11-16

Task created (Ready) after CIP-0015 Accepted; parallel to Stage 1, not a Stage 1 dependency.
