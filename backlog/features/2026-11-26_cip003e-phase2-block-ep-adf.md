---
id: "2026-11-26_cip003e-phase2-block-ep-adf"
title: "Phase 2: block EP (N x K sites), ADF in IVM/PPA/sparse PPA/CGp, capability refusals"
status: "Completed"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-27"
category: "features"
related_cips: ["003E"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- likelihood
- multiclass
---

# Task: Phase 2: block EP (N x K sites), ADF in IVM/PPA/sparse PPA/CGp, capability refusals

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

Generalise CEpInference to N x K diagonal sites via cavityMomentsBlock; ADF engines use the same oracle; IVM selection sums entropy change over the block; refuse likelihoods lacking capabilities.

## Acceptance Criteria

- [x] CEpInference N x K sites; K = 1 reproduces testEp fixtures exactly
- [x] ADF in CIvm/CPpa/CSparsePpa/CGp through cavityMomentsBlock
- [x] IVM block selection reduces to current rule at K = 1
- [x] Capability refusal tests (EP/ADF without CapCavityMoments, Laplace without CapPointDerivs)

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.

### 2026-11-27
Completed. `CEpInference::runBlock` (N x K diagonal sites via `cavityMomentsBlock`/`logZBlock`), `CGp` EP for D > 1 or blockSize > 1 (D = 1 scalar keeps the legacy probit path, so `testEp` fixtures are unchanged), and a block branch in `CNoise::getNuG`, the single choke point for `CIvm`, `CPpa` and `CSparsePpa` ADF. IVM selection already sums the entropy change over the D latents, so it reduces to the current rule at D = 1; no change needed. `CGplvm` takes `CScaleNoise` and `CBayesGplvm`/`CMrd` are Gaussian-only, so they are scalar by type. Laplace refuses blockSize > 1. Convention noted: the probit `CNoise` is a step function on f + eps with the softener as White in K, so CNoise-driven EP on K + I equals the legacy Phi(y f) run() on K (log Z and dL/dK checked). Tests: `testEpBlock`, `testEp` (multi-output Gaussian CGp EP equals the exact GP; refusals), `testNoiseBlock` (ADF block == scalar, bitwise). 15 other no-MATLAB suites pass.

### 2026-11-27 (follow-up)
GPy-convention probit: `CProbitNoise::setGpyMode(true)` sets sigma2 = 1 (Phi(y f) on f, zero bias); the default stays the CIP-0027 convention for IVM/ADF. `CGp::setEpInference` switches a probit to GPy mode and now always uses the block engine, so a non-probit likelihood is no longer silently treated as probit. Probit cavity moments accept {0,1} as well as +-1 targets. In GPy mode the block path equals the legacy `run()` on log Z, posterior and dL/dK. Laplace on `CGp` now supports D outputs for scalar-block likelihoods (one engine per column, shared kernel; column sums checked) and requires `CapPointDerivs`; coupled blocks still refuse until the block-W Laplace of Phase 5. Gap: GPy mode is runtime state and is not written to `.model` streams.
