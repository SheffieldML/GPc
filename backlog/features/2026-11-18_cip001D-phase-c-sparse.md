---
id: "2026-11-18_cip001D-phase-c-sparse"
title: "CIP-001D Phase C: Sparse DTCVAR Python façade"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-19"
category: "features"
related_cips: ["001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip001D-phase-b-gp"]
tags:
- backlog
- python
- sparse
- dtcvar
---

# Task: CIP-001D Phase C sparse DTCVAR

## Description

Expose sparse DTCVAR regression from Python (`SparseGPRegression` or equivalent) with inducing inputs, matching native policy (FITC not default).

## Acceptance Criteria

- [x] Construct sparse model from NumPy `X`, `Y`, `Z` with `dtcvar`
- [x] `optimize` / `predict` / path get/set work
- [x] Pytest vs native or fixed-seed smoke (`python/tests/test_sparse_dtcvar.py`)
- [x] Document DTCVAR-as-default; FITC/DTC available via `approx=`

## Related

- CIP: 001D

## Progress Updates

### 2026-11-18

Task created (Ready) after layering lock for Phase E notebooks.

### 2026-11-19

Implemented `gpc.SparseGPRegression(X, Y, Z)` → `CGp::DTCVAR` with fixed inducing by default; smoke tests green.

### 2026-11-19

Exposed `approx="fitc"` / `"dtc"`; removed stale CLI FITC `exitError` in `gp.cpp`. DTCVAR remains the default.
