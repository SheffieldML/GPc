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

Implement `CArcCosineKern` (orders 0 and 1) with `testKern`-style checks / FD, stream factory, CLI `-k arccos` / `--arccos-order`, and weight-space paths. Order 0 is the public kernel for the \(w,b\to\infty\) MLP limit (simple smoothed units, no learnable \(\Sigma\)); order 1 is smoothed ReLU. Do not fold “infinite MLP” into a `CMlpKern` flag.

## Acceptance Criteria

- [ ] `CArcCosineKern` with order in \(\{0,1\}\); stream type `arccos` / `arccosine`
- [ ] `wsCross*` for order 0 (infinite-\(\Sigma\) / angular) and order 1 (smoothed ReLU)
- [ ] CLI `-k arccos` with `--arccos-order`
- [ ] Gate: large finite \((w,b)\) on `CMlpKern` ws\* approaches order-0 formula
- [ ] FD / smoke tests green

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-1-mlp-isotropic

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted.

### 2026-11-19 (design)
Order 0 designated as infinite-\((w,b)\) home vs finite-\(\Sigma\) `CMlpKern`; literature positioning vs Dutordoir/Sun in CIP-0028.
