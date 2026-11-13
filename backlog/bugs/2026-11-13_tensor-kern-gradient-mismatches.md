---
id: "2026-11-13_tensor-kern-gradient-mismatches"
title: "Tensor kernel gradients disagree with MATLAB fixtures"
status: "Completed"
priority: "Medium"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "bugs"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- testing
- kern
- bugs
---

# Task: Investigate tensor kernel gradient mismatches

## Description

Under `USE_MATIO=1`, `testKern` for `tensor` matches full/double/diag compute and stream I/O against `matfiles/tensorKernTest.mat`, but parameter gradients, X/X2 parameter gradients, X gradients, and diagonal X gradients disagree by large absolute amounts (up to ~1e8). Compound kernels with the same components pass. Likely a CTensorKern gradient bug or a fixture/parameter-transform mismatch rather than matio loading.

## Acceptance Criteria

- [ ] Root-cause gradient gaps vs the matfile generator
- [ ] Fix CTensorKern gradients and/or refresh fixtures
- [ ] Re-enable `testType("tensor")` in `testKern.cpp`
- [ ] CI Reference tests pass with tensor enabled

## Related

- CIP: 0001
- PR: https://github.com/SheffieldML/GPc/pull/15

## Progress Updates

### 2026-11-13

Observed on CI after matio bring-up; tensor temporarily disabled in `testKern` main for the CIP-0001 gate.

### 2026-11-13 (CIP-0003)

Resolved on branch `cip0003-reference-mismatch-backlog`; re-enabled in the Reference tests hard gate.
