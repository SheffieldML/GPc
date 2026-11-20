---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000C"
last_updated: "2026-11-20"
status: "Closed"
compressed: false
related_requirements: ["0005"]
related_cips: ["000B"]
tags:
- cip
- kernels
- gpy
- verification
title: "Finish ARD stubs and import StdPeriodic, Cosine, Brownian"
---

# CIP-000C: Finish ARD stubs and import StdPeriodic, Cosine, Brownian

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [x] Closed - Verified and complete

## Summary

Close the historically commented ARD paths for Matérn 3/2, Matérn 5/2, and rational quadratic; put `CExpKern` on the same verification footing as the other stationary kernels; and add three GPy-aligned `CKern` types that current composition cannot replace: standard periodic, cosine, and Brownian. Multi-output coregionalization is explicitly out of scope (CIP-000D / REQ-0006).

## Stream type strings

| Type string | Class |
| --- | --- |
| `matern32ard` | `CMatern32ardKern` |
| `matern52ard` | `CMatern52ardKern` |
| `ratquadard` | `CRatQuadardKern` |
| `exp` | `CExpKern` (fixture + `testKern` only; already existed) |
| `stdperiodic` | `CStdPeriodicKern` (isotropic period / lengthScale v1) |
| `cosine` | `CCosineKern` |
| `brownian` | `CBrownianKern` (1-D input enforced) |

Brownian requires a scalar time-like input (`inputDim == 1`). Multi-dimensional Brownian is refused.

## Implementation Status

- [x] Matérn/RatQuad ARD classes + factory + tests
- [x] Exponential in `testKern` + fixture
- [x] StdPeriodic + fixture
- [x] Cosine + fixture
- [x] Brownian + fixture
- [x] CLI/README type names
- [x] CI matio path includes new types (`testKern` 0 failures locally)
- [x] Closed after PR CI

## Testing Strategy

- `scripts/gen_cip000C_kern_fixtures.py` — numpy reference + FD grads → `matfiles/<type>KernTest.mat`
- `make USE_MATIO=1 testKern && ./testKern`

## Related Requirements

- [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md) — primary
- [REQ-0002](../requirements/req0002_runnable-verification.md) — verification path

## References

- GPy: `StdPeriodic`, `Cosine`, `Brownian`, Matérn/RatQuad ARD
- Fixture generator note: `fd_grad_X` perturbs `X` (the `getGradX` row), not `X2`
