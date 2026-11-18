---
id: "2026-11-18_cip001C-stability-cells"
title: "CIP-001C Phase 4: stability-suite point vs white-noise cells"
status: "Completed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001C", "0019", "001A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- inducing
- numerics
- stability
---

# Task: CIP-001C Phase 4 — stability suite cells

## Description

Add numerical-stability suite cells (CIP-0019/001A) comparing point vs white-noise inducing on a lengthscale sweep with fixed \(Z\): finiteness, jitter added, \(\kappa(K_{uu})\), ELBO gap, under `stable` and `lq`.

## Acceptance Criteria

- [x] Lengthscale-sweep cell in `numerical_stability/` (`wn_ls_sweep_*`)
- [x] White-noise path finite across the sweep under StableFactor and LQ
- [x] Point path \(\kappa(K_{uu})\) recorded as contrast (≫ WN; kernel-driven)
- [x] ELBO/bound gap vs point inducing recorded per \(\gamma\) and mode
- [x] Metrics emitted in CIP-001A categorised format (`B_linalg`, `F_model`)

## Related

- CIP: 001C
- REQ: 000D

## Progress Updates

### 2026-11-18
Implemented in `testNumericalStability`: WN finite stable/lq; max \(\kappa\) point \(\sim 10^{18}\) vs WN \(\sim 10^{4}\); suite `passed=65 failed=0`.
