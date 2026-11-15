---
id: "0005"
title: "Kernel library covers finished ARD stubs and high-value GPy stationary/periodic primitives"
status: "Validated"
priority: "Medium"
created: "2026-11-15"
last_updated: "2026-11-15"
related_tenets: ["compose-through-interfaces", "assert-and-verify", "ecosystem-interoperability"]
stakeholders: ["maintainers", "modellers"]
tags: ["kernels", "gpy", "covariance"]
---

# REQ-0005: Kernel library covers finished ARD stubs and high-value GPy stationary/periodic primitives

## Description

GPc exposes, through the existing `CKern` contracts and composition (`cmpnd` / `tensor`), the covariance pieces that were historically stubbed for ARD Matérn and rational quadratic, plus a small set of GPy-aligned primitives that cannot be synthesised from the current catalogue: a standard periodic kernel, a cosine kernel, and Brownian motion. Existing kernels already present (including exponential) have a runnable verification path comparable to the other `testKern` types.

**Why this matters**: Compose Through Interfaces expects new covariances as `CKern` implementations; Assert Invariants requires fixtures and gradient checks; Interoperate With the GP Ecosystem means users moving from GPy should find the common stationary and periodic building blocks here without porting the entire GPy catalogue.

**Who benefits**: Modellers needing periodic, oscillatory, or simple non-stationary structure; maintainers closing long-standing commented paths in `testKern` and the stream factory.

## Acceptance Criteria

- [x] Matérn 3/2, Matérn 5/2, and rational quadratic ARD kernels are implemented, stream-loadable, and covered by reference tests
- [x] Exponential kernel is exercised in the reference suite (not only present in source)
- [x] Standard periodic, cosine, and Brownian kernels are implemented as `CKern` types with stream I/O and reference tests
- [x] New types appear in the kernel stream factory with stable type strings
- [x] Out of scope remains explicit: Gibbs, ODE/LFM/SDE, integral-observation, and multi-output coregionalization (see REQ-0006)

## Notes (Optional)

Fixtures may be regenerated from GPy (or an equivalent documented generator) under the matio verification path; MATLAB-era fixtures are not required for brand-new types.

## References

- **Related Tenets**: compose-through-interfaces, assert-and-verify, ecosystem-interoperability
- **External Links**: GPy `StdPeriodic`, `Cosine`, `Brownian`, stationary ARD patterns

## Progress Updates

### 2026-11-15
Requirement created alongside CIP-000C.
