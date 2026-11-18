---
id: "2026-11-18_silent-test-exit-codes"
title: "Reference tests report failures but exit 0 (blind CI gate)"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "bugs"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- bugs
- testing
- ci
- assert-and-verify
---

# Task: Silent test exit codes blind the reference-tests gate

## Description

`testNdlutil` and `testTransform` printed failure counts (`Total failures: 9+`, `Number of failures: 25`) but `main` returned 0 (or fell off the end). `make run-matio-tests` / GitHub Actions `reference-tests.yml` therefore stayed green.

Root causes of the reported failures (not real model bugs):

- `testNdlutil` compared MATLAB-era fixtures against `ndlutil::EPS` (`DBL_EPSILON` ≈ 2e-16); diffs were ~1e-15..2e-14.
- `testTransform` required exact `atox`/`xtoa` invertibility for `|a| > limVal` (36), where transforms intentionally clamp; also used exact `!=` for floating round-trips; gradient FAIL lines did not increment `fail`.

Same exit-code hole existed in `testIvm` and `testMltools` (not on the matio gate). Expanding transform probes also exposed `CNegLogLogitTransform::atox` leaving `x=0` for `a >= limVal` instead of softplus saturation `x=a`.

## Acceptance Criteria

- [x] `testNdlutil` / `testTransform` / `testIvm` / `testMltools` return non-zero when `fail > 0`
- [x] `testNdlutil` uses `ndlutil::MATCHTOL` for fixture/constant comparisons
- [x] `testTransform` allows `limVal` clamping, uses MATCHTOL round-trips, counts gradient failures
- [x] `CNegLogLogitTransform::atox` saturates for large positive `a`
- [x] Local `./testNdlutil` and `./testTransform` exit 0 with zero reported failures

## Related

- Tenet: assert-and-verify
- CI: `.github/workflows/reference-tests.yml` → `make run-matio-tests`
- Code: `testNdlutil.cpp`, `testTransform.cpp`, `testIvm.cpp`, `testMltools.cpp`, `CTransform.cpp`

## Progress Updates

### 2026-11-18

External review flagged the blind gate. Fixed exit codes, tolerances, clamp-aware transform tests, and softplus upper saturation. Marked Completed.
