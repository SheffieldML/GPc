---
id: "2026-11-28_cip0043-phase1-batched-inducing-grads"
title: "CIP-0043 Phase 1: batched inducingGradX / inducingGradZ for ReLU kernels"
status: "Proposed"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0043", "003F"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_cip0043-phase0-relu-kernel"]
tags:
- backlog
- kernel
- relu
- performance
---

# Task: CIP-0043 Phase 1 — batched inducing gradients

## Description

Override `inducingGradX` and `inducingGradZ` for `CReluKern` and `CReluardKern` in GEMM form, with the cached \(K_{uf}\) overload, following `mlpFirstArgGrad` and `mlpInducingGradZ`.

## Acceptance Criteria

- [ ] Both kernels added to `testInducingGradZ`; batched result matches the base-class `getGradX` loop at 1e-12, with and without cached \(K_{uf}\).
- [ ] `inducingGradZ` includes the \(K_{uf}\), symmetric \(K_{uu}\) and diagonal terms.
- [ ] `testInducingGradZ` and `testSvgp` pass.

## Related

- CIP: 0043, 003F

## Progress Updates

### 2026-11-28

Task created, Proposed (blocked on Phase 0).
