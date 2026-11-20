---
id: "2026-11-20_cip0030-phase3-probit-ordered"
title: "CIP-0030 Phase 3: probit / ordered on CSviBayesGplvm"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "002A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0030-phase2-inducing-families"]
tags:
- backlog
- svgp
- svi
- bayesian-gplvm
- probit
- ordered
- gauss-hermite
---

# Task: CIP-0030 Phase 3 — probit / ordered Bayesian GP-LVM

## Description

Wire non-Gaussian likelihoods through `CSviBayesGplvm::setLikelihood` (delegating to `CSvgp`). Implement `COrderedNoise::expectedLogLikelihood` via Gauss–Hermite. Fix `ndlutil::gaussHermiteExpect` ∂/∂var (was off by ½). Verify V4 FD and N=2 probit orthant bound.

## Acceptance Criteria

- [x] `setLikelihood` / `getLikelihood` on `CSviBayesGplvm`
- [x] Ordered GH `expectedLogLikelihood` (bias via `setExpectedLogLikBias`)
- [x] GH `gVar` matches FD; CSvgp / SVI global grads correct for probit
- [x] V4 FD for probit and ordered at prior \(q(\mathbf v)\)
- [x] N=2 probit: SVI ELBO+KL\((q(\mathbf X))\) ≤ exact orthant log marginal
- [x] Ordered natgrad increases ELBO; `testSviBayesGplvm` and `testSvgp` green

## Related

- CIP: 0030

## Progress Updates

### 2026-11-20

Phase 3 implemented and verified. Also fixed pre-existing GH `gVar` factor-of-two bug.
