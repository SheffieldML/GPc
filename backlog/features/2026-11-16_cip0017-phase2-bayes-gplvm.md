---
id: "2026-11-16_cip0017-phase2-bayes-gplvm"
title: "Phase 2: CBayesGplvm ELBO, grads, stream, tests"
status: "Completed"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "features"
related_cips: ["0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-16_cip0017-phase1-psi-stats"]
tags:
- backlog
- gplvm
- bayesian
- variational
---

# Task: Phase 2 Bayesian GP-LVM model

## Description

Implement `CBayesGplvm`: collapsed Titsias / GPy VarDTC bound with psi statistics, minus \(\mathrm{KL}(q(\mathbf{X})\|p(\mathbf{X}))\), with analytic gradients, `checkGradients`, stream I/O, and `testBayesGplvm`.

## Acceptance Criteria

- [x] ELBO matches GPy VarDTC structure (spherical beta) minus diagonal-Gaussian KL
- [x] `checkGradients` green on a tiny problem
- [x] Stream round-trip for μ, S, Z, beta (kernel block written/consumed; caller retains `pkern`)
- [x] `make testBayesGplvm` wired and green
- [x] Thin CLI demo `demo_bayes_gplvm`

## Related

- CIP: 0017

## Progress Updates

### 2026-11-16

Task opened; Phase 1 PR #50 up; implementation starting on `cip0017-bayesian-gplvm`.

### 2026-11-16 (later)

`CBayesGplvm` + `testBayesGplvm` green (maxAbs FD ≈ 1e-8). Fixed upper-Cholesky `backsub` orientation, `logDet(B)` factor, and `∂L/∂β` via GPy `dL_dR`→`σ²` chain. Demo `examples/demo_bayes_gplvm.cpp` optimises a toy ELBO.
