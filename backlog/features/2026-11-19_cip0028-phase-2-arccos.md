---
id: "2026-11-19_cip0028-phase-2-arccos"
title: "CIP-0028 Phase 2: arc-cosine kernel + weight-space path"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0028-phase-1-mlp-isotropic"]
tags:
- backlog
- sparse
- inducing
- weight-space
- arccos
---

# Task: CIP-0028 Phase 2 — arc-cosine + weight-space

## Description

Implement `CArcCosineKern` (orders 0 and 1) with `testKern`-style checks / FD, stream factory, CLI `-k arccos` / `--arccos-order`, and weight-space `wsCross*` for order 1 (smoothed ReLU); order 0 if the integral is cheap.

## Acceptance Criteria

- [ ] `CArcCosineKern` with order in \(\{0,1\}\`; stream type `arccos` / `arccosine`
- [ ] `wsCross*` path for order 1; order 0 if feasible
- [ ] CLI `-k arccos` with `--arccos-order`
- [ ] FD / smoke tests green

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-1-mlp-isotropic

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted.
