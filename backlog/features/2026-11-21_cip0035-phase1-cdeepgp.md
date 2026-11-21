---
id: "2026-11-21_cip0035-phase1-cdeepgp"
title: "CIP-0035 Phase 1: CDeepGp Gaussian + inducing points"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0035"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0035-phase0b-layer-api"]
tags:
- backlog
- deep-gp
- dsvi
- inducing-points
---

# Task: CIP-0035 Phase 1 — `CDeepGp` (Gaussian; points)

## Description

Implement `CDeepGp : CStochasticOptimisable` with identity/PCA mean functions, per-layer variance \(\chi/(L-1)\) on hidden layers, inducing points, Gaussian final layer, and seeded minibatch DSVI. Gates V1–V6 from CIP-0035.

## Acceptance Criteria

- [x] `CDeepGp` with \(L\ge1\), inducing points, Gaussian likelihood
- [x] V1: \(L=1\) equals `CSvgp` ELBO (round-off)
- [x] V2: deterministic hidden layer reduces to `CSvgp` on \(\mathbf X P_1\)
- [x] V3: two-layer psi exactness within Monte Carlo error
- [x] V4–V6: FD, minibatch exactness, seed reproducibility
- [x] `testDeepGp` green (matio-free)

## Related

- CIP: 0035

## Progress Updates

### 2026-11-21
Implemented `CDeepGp` / `testDeepGp`. Fixed double `/K` in multi-sample `objectiveGradBatch` (c already includes \(1/K\)). V1–V6 green; `testSvgp` still green.
