---
id: "2026-11-18_cip0024-eq-core-factory-paths"
title: "CIP-0024: CEqKern + factory + ExpQuad lengthscale paths"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["0024"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- kernels
- gpy
- cip0024
---

# Task: CEqKern + factory + ExpQuad lengthscale paths

> Implements [CIP-0024](../../cip/cip0024.md).

## Description

Add `CEqKern` as a thin subclass of `CRbfKern` (`type: eq`, `gpyComponentName` → `ExpQuad`), wire stream factory read aliases (`eq` / `ExpQuad` / `expquad`), and ensure CIP-0014 path conversion treats `eq` like RBF for `lengthscale` ↔ `inverseWidth`.

## Acceptance Criteria

- [x] `CEqKern` constructs, clones, and writes stream type `eq`
- [x] `readKernFromStream` accepts `eq`, `ExpQuad`, `expquad`
- [x] `ExpQuad.lengthscale` / `ExpQuad.variance` get/set in GPy units
- [x] Setting \(\ell\) yields same \(K\) as `CRbfKern` with \(w=1/\ell^2\) within `MATCHTOL`

## Implementation Notes

Reuse `LEAF_RBF_LENGTHSCALE`; do not duplicate `K_of_r`. No `eqard`.

## Related

- CIP: 0024
- Requirement: REQ-0012

## Progress Updates

### 2026-11-18

Task created when CIP-0024 was Accepted.

### 2026-11-18
Completed on `cip0024-eq-expquad`.
