---
id: "2026-11-21_cip0035-phase0b-layer-api"
title: "CIP-0035 Phase 0b: marginalsAt / marginalsGradAt / klQ layer API"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0035", "0030"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0035-phase0a-cholesky-qv"]
tags:
- backlog
- deep-gp
- svgp
- layer-api
---

# Task: CIP-0035 Phase 0b — layer API

## Description

Add `marginalsAt`, `marginalsGradAt`, and `klQ` to `CSvgp` for hidden-layer stacking. `marginalsGradAt` reuses the existing `elboGradAt` path with caller-supplied \((\bar{\boldsymbol\mu},\bar{\mathbf v})\) in place of the likelihood's \((g^\mu,g^v)\).

## Acceptance Criteria

- [x] `marginalsAt` / `marginalsGradAt` / `klQ` on `CSvgp`
- [x] FD for all inducing families; `marginalsGradAt` vs FD on `marginalsAt`
- [x] `elboGradAt` reproduced by chaining `marginalsGradAt` with likelihood \((g^\mu,g^v)\) (round-off)
- [x] `testSvgp` covers the new API

## Related

- CIP: 0035

## Progress Updates

### 2026-11-21 (complete)

Phase 0b landed: `marginalsAt` / `marginalsGradAt` / `klQ`; FD on gX for points/WN/weightspace/greens; `elboGradAt` gX/gKern match chained likelihood grads (round-off 0); `klQ` FD-green. `./testSvgp` 0 failures.
