---
id: "2026-11-28_cip0043-phase2-weightspace-hermite"
title: "CIP-0043 Phase 2: weight-space and Hermite hooks, compound dispatch"
status: "Proposed"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0043", "0028", "0032", "003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_cip0043-phase1-batched-inducing-grads"]
tags:
- backlog
- kernel
- relu
- weight-space
- inducing
---

# Task: CIP-0043 Phase 2 — weight-space and Hermite

## Description

Implement `wsCrossCompute`, `wsGrad{Params,W,X,S}` and `wsRidge*` for ReLU kernels with general \(\Sigma\), sharing `RidgeBulk` and `arccosSoftReLU`. Extend `isWeightSpaceKern` and the compound dispatcher so `relu` / `reluard` plus optional `white` or `bias` work for `--inducing weightspace` and Hermite windows. Add `CKern::supportsRidgeWindows()` (default false) for use in Phase 3.

## Acceptance Criteria

- [ ] Ridge \(K_{fu}\) and all five gradients match the per-element reference at 1e-12 (`CKern::setLegacyRidge(true)` path).
- [ ] With \(\sigma_w=\sigma_b=1\), results match `CArcCosineKern` order 1 up to the documented constant.
- [ ] Hermite quadrature and gradient gates from CIP-003D pass for ReLU.
- [ ] `relu + white` compound accepted; unsupported compounds still raise typed errors.
- [ ] `supportsRidgeWindows()` true for MLP, MLP-ARD, ArcCosine, ReLU, ReLU-ARD and compounds of one of these.

## Related

- CIP: 0043, 0028, 0032, 003D

## Progress Updates

### 2026-11-28

Task created, Proposed (blocked on Phase 1).
