---
id: "2026-11-13_matrix-lapack-reference-mismatches"
title: "testMatrix mismatches vs MATLAB fixtures on modern BLAS"
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
- lapack
- bugs
---

# Task: Investigate testMatrix LAPACK mismatches

## Description

With libmatio loading the existing `matfiles/` fixtures on Ubuntu CI, `testMatrix` reports failures (and can throw `std::out_of_range` from `BOUNDCHECK`) on some paths: `syev` eigenvector compare, several `trsm` cases, and a later bound check after `symv`. Many other BLAS paths match. This is separate from matio enablement — fixture load for gemm/syrk/trmm/etc. works.

## Acceptance Criteria

- [ ] Root-cause each failing case (sign ambiguity, triangular side/unit flags, index bounds, BLAS library difference)
- [ ] Either fix CMatrix wrappers, update tolerances/comparisons, or refresh fixtures with documented generator
- [ ] `testMatrix` exits 0 on CI under `USE_MATIO=1`
- [ ] Reference-tests workflow requires `testMatrix` again

## Related

- CIP: 0001
- PR: https://github.com/SheffieldML/GPc/pull/15

## Progress Updates

### 2026-11-13

Observed on CI after matio bring-up; `testMatrix` temporarily `continue-on-error` in the Reference tests workflow.

### 2026-11-13 (CIP-0003)

Resolved on branch `cip0003-reference-mismatch-backlog`; re-enabled in the Reference tests hard gate.
