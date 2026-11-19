---
id: "2026-11-19_cip0028-phase-1-mlp-isotropic"
title: "CIP-0028 Phase 1: MLP isotropic weight-space K_fu closed forms"
status: "Proposed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0028-phase-0-skeleton"]
tags:
- backlog
- sparse
- inducing
- weight-space
- mlp
---

# Task: CIP-0028 Phase 1 — MLP isotropic weight-space

## Description

Implement closed-form \(K_{fu}\) for `CMlpKern` with Gaussian windows on weight space (erf / Williams MLP), plus gradients w.r.t. kernel params, weight centres \(W\), window width \(s\), and data \(X\). Wire through `CInducingWeightSpace` so DTC/DTCVAR sparse paths run end-to-end for MLP + weight-space.

## Acceptance Criteria

- [ ] Closed-form `wsCross*` + grads on `CMlpKern`
- [ ] MC / numeric convolution matches closed form on 1-D and small \(d\)
- [ ] FD agreement on kernel params, \(W\), and \(s\) / \(X\) as scoped
- [ ] `testCGp` DTCVAR smoke with weight-space + MLP
- [ ] \(\kappa(K_{uu})\) independent of `weightVariance` on a small sweep cell (Phase 3)

## Implementation Notes

Pin constants against existing `CMlpKern::computeElement` and CIP-0028 erf formula. Do not change point-inducing MLP semantics.

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-0-skeleton

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted; blocked on Phase 0.
