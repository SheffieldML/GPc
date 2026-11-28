---
id: "2026-11-28_cip0043-phase0-relu-kernel"
title: "CIP-0043 Phase 0: CReluKern and CReluardKern, factory, testKern"
status: "Ready"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0043"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- kernel
- relu
- mlp
---

# Task: CIP-0043 Phase 0 — ReLU kernel, factory, gradients

## Description

Add `CReluKern` and `CReluardKern` with the `CMlpKern` parameter set (`weightVariance`, `biasVariance`, `variance`; per-dimension weight variances for ARD). Register `relu` / `reluard` in the type/name tables and the stream factory. Parameterisation, class layout and the `ArcCosine` relationship follow the decisions in CIP-0043.

## Acceptance Criteria

- [ ] `compute` (three overloads), `computeElement`, `diagComputeElement`, `getGradParams` (both overloads), `getGradX`, `getDiagGradX`, `clone`, versioned stream read/write.
- [ ] Factory and name tables register `relu`, `reluard`; parameter paths (`relu.weightVariance`, …) resolve.
- [ ] `testKern` covers both kernels: symmetry, PSD up to round-off, finite-difference gradients including θ≈0 and θ≈π.
- [ ] Monte Carlo wide-ReLU-network oracle agrees with the closed form within the Monte Carlo standard error.
- [ ] `relu(1,1,1)` equals `arccos` order 1 up to the documented constant.
- [ ] Stream round-trip and `equals` at `ndlutil::MATCHTOL`.
- [ ] `make` and `testKern` pass.

## Implementation Notes

Reuse `arccosClampCos` and the `J_0 = π − θ` form of the derivative from `CArcCosineKern::getGradX`. Document the flat direction in `(weightVariance, biasVariance, variance)` in the class comment.

## Related

- CIP: 0043

## Progress Updates

### 2026-11-28

Task created, Ready.
