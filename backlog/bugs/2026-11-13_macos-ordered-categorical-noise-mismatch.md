---
id: "2026-11-13_macos-ordered-categorical-noise-mismatch"
title: "Ordered categorical noise mismatches on macOS OpenBLAS"
status: "Completed"
priority: "Low"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "bugs"
related_cips: ["0001", "0006"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- bugs
- macos
- noise
- numerics
---

# Task: Ordered categorical noise mismatches on macOS OpenBLAS

## Description

On Apple Silicon with Homebrew OpenBLAS / GCC 16, `testNoise` reported failures for Ordered Categorical log likelihood and parameter / mu / vs gradients against the MATLAB fixtures. Gaussian, Probit, and Null Category passed; `testKern`, `testMatrix`, and `testDist` passed. Linux Reference tests CI remained the hard gate until this was fixed.

## Acceptance Criteria

- [x] Root cause identified (math library / `erfcx` / OpenBLAS vs reference LAPACK, or fixture tolerance)
- [x] Either tolerances adjusted with justification, or numerics fixed so macOS matches Linux gate
- [x] Documented in README or CIP notes if platform-specific soft mismatch remains intentional

## Related

- CIP: 0001
- CIP: 0006
- REQ: 0002

## Progress Updates

### 2026-11-13

Observed while validating `make.osx` matio suite after REQ-0001 macOS build path landed.

Root cause: missing ordered targets are stored as NaN. `COrderedNoise` cast the target to `int` before testing `isnan`. On x86_64, `(int)NaN` is typically `INT_MIN`, so control fell through to the missing-data branch. On ARM64, `(int)NaN` is `0`, so the three missing points in `orderedNoiseTest.mat` were scored as category 0 (Δ log-likelihood ≈ 16.5; bias gradients for processes 2 and 4). Fix: test `isnan` before the cast in `getGradParams`, `getGradInputs`, `likelihoods`, and `logLikelihood`. macOS Reference tests promoted to a hard gate.
