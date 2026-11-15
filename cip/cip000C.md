---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000C"
last_updated: "2026-11-15"
status: "Proposed"
compressed: false
related_requirements: ["0005"]
related_cips: []
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
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Close the historically commented ARD paths for Matérn 3/2, Matérn 5/2, and rational quadratic; put `CExpKern` on the same verification footing as the other stationary kernels; and add three GPy-aligned `CKern` types that current composition cannot replace: standard periodic, cosine, and Brownian. Multi-output coregionalization is explicitly out of scope (CIP-000D / REQ-0006).

## Motivation

`testKern.cpp` and `readKernFromStream` already name `matern32ard`, `matern52ard`, `ratquadard`, and periodic/Gibbs variants, but those classes were never finished. GPy’s public catalogue is much larger; most of it should stay out of GPc. The high-value remainder for single-output modelling is: finish the ARD stubs that match existing `CArdKern` patterns, verify exponential, and bring `StdPeriodic`, `Cosine`, and `Brownian` so periodic and simple non-stationary structure are available under Compose Through Interfaces without bloating the library.

## Detailed Description

### Scope (in)

| Item | GPc target | Notes |
|------|------------|--------|
| Matérn 3/2 ARD | `CMatern32ardKern` | Follow `CRbfardKern` / `CArdKern` |
| Matérn 5/2 ARD | `CMatern52ardKern` | Same |
| RatQuad ARD | `CRatQuadardKern` | Same |
| Exponential | existing `CExpKern` | Add `testType("exp")` + fixture |
| StdPeriodic | `CStdPeriodicKern` (name TBD) | GPy `StdPeriodic` formula; prefer this over resurrecting only `CRbfPeriodicKern` |
| Cosine | `CCosineKern` | Stationary; GPy `Cosine` |
| Brownian | `CBrownianKern` | Non-stationary; \(k(x,x')=\sigma^2 \min(x,x')\) on 1-D index (document multi-D policy) |

### Scope (out)

- Gibbs / Gibbs-periodic / Gibbs-ARD (heavy, historically stubbed; revisit only with a concrete user)
- GPy duplicates of existing kernels (`ExpQuad` ≡ RBF, `OU` ≡ Exp)
- `PeriodicMatern*`, `Sinc`, `ExpQuadCosine`, Wiener velocity, spline, trunc-linear, basis-function kernels
- Entire multi-output stack (`Coregionalize`, IndependentOutputs, …) → **CIP-000D**
- SDE / ODE / LFM / integral-observation kernels

### Design choices

1. **ARD pattern** — Keep GPc’s separate `*ard` classes (not GPy’s `ARD=True` flag on a single class). Reuse `CArdKern` lengthscale storage and the same parameter transforms as `CRbfardKern`. Implement `updateX` where distances can be prefetched cheaply (Matérn/RatQuad), matching Flexibility Without a Performance Tax.

2. **Periodic** — Implement GPy’s `StdPeriodic` (sin-of-scaled-delta form from Rasmussen & Williams / MacKay), not the older commented `rbfperiodic` name alone. Stream type string: prefer `stdperiodic` (document alias if we also accept `rbfperiodic` for old notes). Default: shared period and lengthscale per input dim unless ARD flags are cheap to add in a second pass; v1 may ship isotropic period/lengthscale if that matches fixture simplicity, with ARD as a follow-on backlog item after Accept.

3. **Cosine** — Thin stationary kernel on scaled distance; natural compound partner with RBF/Matérn.

4. **Brownian** — Document as requiring a scalar time-like input (or a designated active dimension). Multi-dimensional Brownian is not a free generalisation; refuse or define an explicit product policy in the class docs and tests.

5. **Verification** — New and previously untested types get `matfiles/<type>KernTest.mat` (or successor) generated from GPy under a documented script in-tree. Gradients via existing `testKern` / `checkGradients` paths. Wire factory branches that are currently commented.

6. **CLI** — Extend `gp` / `ivm` / `ppa` / `gplvm` kernel name lists only where those tools already enumerate kernel types; do not invent a new DSL.

### Alternatives considered

- **Port GPy’s full stationary set** — Rejected; most types are rare and verification cost dominates.
- **Only finish ARD, skip periodic/cosine/Brownian** — Rejected; those three are the “cannot compose from what we have” set.
- **Revive Gibbs instead of Brownian** — Rejected for this CIP; Gibbs is higher complexity and was never the GPy default teaching kernel.

## Implementation Plan

1. **ARD completion**
   - Implement `CMatern32ardKern`, `CMatern52ardKern`, `CRatQuadardKern` in `CKern.h` / `CKern.cpp`
   - Enable factory + `testKern` branches; generate fixtures from isotropic Matérn/RatQuad + ARD lengthscales in GPy
2. **Exponential verification**
   - Add `testType("exp")`; generate `expKernTest.mat`; confirm stream round-trip
3. **StdPeriodic**
   - Implement compute, diag, GradX, GradParams, stream I/O; fixture from GPy `StdPeriodic`
4. **Cosine**
   - Same contract surface; fixture from GPy `Cosine`
5. **Brownian**
   - Same; fixture from GPy `Brownian`; document input constraints
6. **Integration polish**
   - CLI name strings where applicable; brief README note listing new kernel type names
   - Ensure Reference tests / matio CI include the new `testKern` types

## Backward Compatibility

- Additive only: new stream type strings; no change to existing kernel parameter layouts
- Commented factory branches become live; old `.model` files without those types are unaffected
- If `stdperiodic` is chosen over `rbfperiodic`, document the type string in release notes so scripts do not guess

## Testing Strategy

- Extend `testKern` for: `matern32ard`, `matern52ard`, `ratquadard`, `exp`, `stdperiodic` (or chosen name), `cosine`, `brownian`
- Compare K, diag, parameter gradients, and X gradients against generated fixtures under `make USE_MATIO=1 matio-tests`
- Spot-check compound use: e.g. `stdperiodic + rbf + white`, `cosine + matern32 + bias`
- Do not weaken tolerances to land green; fix kernels or regenerate fixtures with a recorded seed/script

## Related Requirements

- [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md) — primary
- Touches verification culture of [REQ-0002](../requirements/req0002_runnable-verification.md) but does not redefine it

## Implementation Status

- [ ] Matérn/RatQuad ARD classes + factory + tests
- [ ] Exponential in `testKern` + fixture
- [ ] StdPeriodic + fixture
- [ ] Cosine + fixture
- [ ] Brownian + fixture
- [ ] CLI/README type names
- [ ] CI matio path includes new types

## References

- GPy: `GPy.kern.StdPeriodic`, `Cosine`, `Brownian`, `Matern32`/`Matern52`/`RatQuad` with `ARD=True`
- In-tree stubs: `testKern.cpp` commented blocks; `readKernFromStream` in `CKern.cpp`
- Tenets: Compose Through Interfaces; Assert Invariants and Verify Against Reference; Flexibility Without a Performance Tax
