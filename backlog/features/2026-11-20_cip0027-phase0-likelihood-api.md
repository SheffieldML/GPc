---
id: "2026-11-20_cip0027-phase0-likelihood-api"
title: "CIP-0027 Phase 0: batched likelihood naming / API sketch"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0027"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- noise
- likelihood
- architecture
---

# Task: CIP-0027 Phase 0 — Batched likelihood API sketch

## Description

Document and sketch the architecture rule-2 likelihood surface on `CNoise` / the Gaussian path: canonical names (`log_density`, `variational_expectations`, `log_predictive_density`, …), batched contiguous helpers, and **no new data or site fields** on likelihoods. Prepare Gaussian helpers Phase A needs for Gram noise (`σ²` on the diagonal).

## Acceptance Criteria

- [x] Batched Gaussian helpers exist (oracles vs existing per-element paths agree within `MATCHTOL`)
- [x] API naming documented (header comments and/or `cip/cip0027/` note)
- [x] Diff adds **zero** new per-point / site fields on `CNoise` subclasses
- [x] Unit test runs without MATLAB fixtures where possible

## Implementation Notes

Keep existing C++ method names as wrappers. Do not migrate IVM/PPA/ADF callers yet (follow-on CIP).

## Related

- CIP: 0027
- Documentation: `docs/target-architecture.md`, `cip/cip0027.md`, `cip/cip0027/phase0_likelihood_api.md`

## Progress Updates

### 2026-11-20

Task created on Accept; Phase 0 started on branch `cip0027-likelihood-owned-noise`.

### 2026-11-20 (done)

`ndlnoise::{addObservationNoiseDiag,gaussianLogDensity,gaussianVariationalExpectations,gaussianLogPredictiveDensity}`; `CGaussianNoise::{setSigma2,getVariance,setVariance}`; `testNoiseBatched` green (0 failures).
