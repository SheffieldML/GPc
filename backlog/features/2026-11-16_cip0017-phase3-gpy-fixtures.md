---
id: "2026-11-16_cip0017-phase3-gpy-fixtures"
title: "Phase 3: GPy text fixtures + C++ compare for Bayesian GPLVM"
status: "Completed"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "features"
related_cips: ["0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-16_cip0017-phase2-bayes-gplvm"]
tags:
- backlog
- gplvm
- bayesian
- fixtures
- gpy
---

# Task: Phase 3 GPy parity fixtures

## Description

Commit a maintainer GPy script that dumps psi, ELBO, and natural-space gradients for a seeded tiny Bayesian GP-LVM problem, plus a C++ compare path in `testBayesGplvm` that loads the fixture without needing GPy at CI runtime.

## Acceptance Criteria

- [x] `scripts/gen_cip0017_bayes_gplvm_fixtures.py` regenerates `testdata/cip0017/bayes_gplvm_rbf_tiny.txt` when GPy is available
- [x] `testBayesGplvm` compares psi0/1/2, ELBO, and natural grads (tol 1e-6; observed ~1e-15)
- [x] `CBayesGplvm` jitter matches GPy VarDTC (`1e-8`)
- [x] CIP-0015 / REQ-0009 already defer Bayesian GPLVM to CIP-0017 / REQ-000B

## Related

- CIP: 0017
- PR: #51

## Progress Updates

### 2026-11-16

Fixture + compare green; maxGradDiff ≈ 3.6e-15 vs GPy 1.15.0.
