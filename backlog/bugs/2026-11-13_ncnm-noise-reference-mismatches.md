---
id: "2026-11-13_ncnm-noise-reference-mismatches"
title: "Null-category noise disagrees with MATLAB fixtures"
status: "Proposed"
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
- noise
- bugs
---

# Task: Investigate ncnm noise reference mismatches

## Description

Under `USE_MATIO=1`, `testNoise` for `ncnm` (Null Category) loads `matfiles/ncnmNoiseTest.mat` and reports failures against the stored MATLAB reference values: log-likelihood (~17 absolute), parameter gradient (max abs diff ~45), mu gradient (~45), and vs gradient (max abs diff ~2e5). Gaussian, probit, and ordered noise match; stream round-trip for ncnm itself passes. This looks like a numeric/algorithm mismatch rather than fixture I/O.

## Acceptance Criteria

- [ ] Root-cause the likelihood and gradient gaps vs the matfile generator
- [ ] Fix CNcnmNoise and/or refresh fixtures with a documented generator
- [ ] Re-enable `testType("ncnm")` in `testNoise.cpp`
- [ ] CI Reference tests pass with ncnm enabled

## Related

- CIP: 0001
- PR: https://github.com/SheffieldML/GPc/pull/15

## Progress Updates

### 2026-11-13

Observed on CI after matio bring-up; `ncnm` temporarily disabled in `testNoise` main so the CIP-0001 gate can stay focused on loaders and the noise types that already match.
