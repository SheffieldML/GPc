---
id: "2026-11-19_cip002A-phase4-non-gaussian"
title: "CIP-002A Phase 4: expectedLogLikelihood + Gauss-Hermite for probit/ordered"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["002A", "0026", "0027"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002A-phase3-adam"]
tags:
- backlog
- svgp
- probit
- gauss-hermite
---

# Task: CIP-002A Phase 4 — non-Gaussian expected log-likelihood

## Description

Add `CNoise::expectedLogLikelihood(y, mu, var, gMu, gVar)` distinct from ADF `logLikelihood(mu, var, y)`. Closed form for Gaussian; Gauss–Hermite (default 20 nodes) for probit and ordered categorical. Wire into `CSvgp` with \(\gamma\) warm-up. Compare against exact probit orthant marginal and sparse PPA on shared problems; GPy `SVGP` fixture when available. NCNM out of scope.

## Acceptance Criteria

- [ ] Probit ELBO ≤ exact orthant log marginal on small problems
- [ ] Gaussian path unchanged vs Phase 1/2 oracles
- [ ] GPy fixture parity or documented deferral with a tracked follow-up

## Related

- CIP: 002A
- REQ: 0015

## Progress Updates

### 2026-11-19
Task created on CIP acceptance.

### 2026-11-19
Marked Completed after `testSvgp` verification on `cip002A-svgp`.
