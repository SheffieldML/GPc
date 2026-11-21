---
id: "2026-11-21_cip0033-phase-c-cgp-laplace"
title: "CIP-0033 Phase C: Laplace on CGp FTC"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0033"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0033-phase-b-cgp-ep"]
tags:
- backlog
- laplace
- cgp
- gpy
---

# Task: CIP-0033 Phase C — `CGp` FTC + Laplace

## Description

Ship GPy-mirrored Laplace approximation on dense `CGp` for Bernoulli–probit: mode finding, approximate marginal, kernel grads (explicit + implicit), fixture vs numpy twin, and agreement with EP on a shared toy.

## Acceptance Criteria

- [x] `CGp::setLaplaceInference(true)` FTC path
- [x] Mode + Hessian oracles (`bernoulliProbitLaplaceDerivs`) with FD check
- [x] Kernel grads vs FD; optimise improves
- [x] Fixture `cip/cip0033/laplace_ftc_fixture.txt`; |L_lap − L_ep| small on shared 4-point toy
- [x] Tutorial updated for EP vs Laplace

## Related

- CIP: 0033
- REQ: 0019

## Progress Updates

### 2026-11-21

Phase C delivered on branch `cip0033-likelihood-inference-ep`.
