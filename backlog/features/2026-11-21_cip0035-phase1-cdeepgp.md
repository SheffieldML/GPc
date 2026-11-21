---
id: "2026-11-21_cip0035-phase1-cdeepgp"
title: "CIP-0035 Phase 1: CDeepGp Gaussian + inducing points"
status: "Ready"
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

- [ ] `CDeepGp` with \(L\ge1\), inducing points, Gaussian likelihood
- [ ] V1: \(L=1\) equals `CSvgp` ELBO (round-off)
- [ ] V2: deterministic hidden layer reduces to `CSvgp` on \(\mathbf X P_1\)
- [ ] V3: two-layer psi exactness within Monte Carlo error
- [ ] V4–V6: FD, minibatch exactness, seed reproducibility
- [ ] `testDeepGp` green (matio-free)

## Related

- CIP: 0035
