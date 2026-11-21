---
id: "2026-11-21_cip0035-phase0b-layer-api"
title: "CIP-0035 Phase 0b: marginalsAt / marginalsGradAt / klQ layer API"
status: "Ready"
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

- [ ] `marginalsAt` / `marginalsGradAt` / `klQ` on `CSvgp`
- [ ] FD for all inducing families; `marginalsGradAt` vs FD on `marginalsAt`
- [ ] `elboGradAt` reproduced by chaining `marginalsGradAt` with likelihood \((g^\mu,g^v)\) (round-off)
- [ ] `testSvgp` covers the new API

## Related

- CIP: 0035
