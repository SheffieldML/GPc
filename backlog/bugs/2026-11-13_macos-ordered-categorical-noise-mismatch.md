---
id: "2026-11-13_macos-ordered-categorical-noise-mismatch"
title: "Ordered categorical noise mismatches on macOS OpenBLAS"
status: "Proposed"
priority: "Low"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "bugs"
related_cips: ["0001"]
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

On Apple Silicon with Homebrew OpenBLAS / GCC 16, `testNoise` reports failures for Ordered Categorical log likelihood and parameter / mu / vs gradients against the MATLAB fixtures. Gaussian, Probit, and Null Category pass; `testKern`, `testMatrix`, and `testDist` pass. Linux Reference tests CI remains the hard gate.

## Acceptance Criteria

- [ ] Root cause identified (math library / `erfcx` / OpenBLAS vs reference LAPACK, or fixture tolerance)
- [ ] Either tolerances adjusted with justification, or numerics fixed so macOS matches Linux gate
- [ ] Documented in README or CIP notes if platform-specific soft mismatch remains intentional

## Related

- CIP: 0001
- REQ: 0002

## Progress Updates

### 2026-11-13

Observed while validating `make.osx` matio suite after REQ-0001 macOS build path landed.
