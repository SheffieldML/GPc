---
id: "2026-11-18_cip001D-phase-c-sparse"
title: "CIP-001D Phase C: Sparse DTCVAR Python façade"
status: "Ready"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
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

- [ ] Construct sparse model from NumPy `X`, `Y`, `Z` with `dtcvar`
- [ ] `optimize` / `predict` / path get/set work
- [ ] Pytest vs native or fixed-seed smoke
- [ ] Document FITC-not-default

## Related

- CIP: 001D

## Progress Updates

### 2026-11-18

Task created (Ready) after layering lock for Phase E notebooks.
