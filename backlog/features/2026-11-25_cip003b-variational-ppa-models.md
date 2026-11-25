---
id: "2026-11-25_cip003b-variational-ppa-models"
title: "CIP-003B: HMC for variational and PPA models"
status: "Completed"
priority: "Medium"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["003B"]
owner: "Neil Lawrence"
dependencies: ["2026-11-25_cip003b-lengthscale-prior-units"]
tags:
- backlog
- hmc
- variational
- ppa
---

# Task: HMC over hyperparameters of variational and PPA models

## Description

CIP-003B Phase 4 left the variational and PPA models open "after a check of each objective". The check, model by model:

| Model | Objective | Deterministic with exact gradient? | Decision |
|---|---|---|---|
| `CBayesGplvm`, `CMrd` | VarDTC bound over kernel, beta, q(X), Z | Yes | Supported: sample kernel parameters and `likelihood.variance` with q(X), Z fixed (`CBoundHyperTarget`) |
| `CPpa` | M-step objective `kernelLogLikelihood` at fixed sites | Yes (exact gradient) | Supported on request (`setKernelObjective(true)`); outer `approxLogLikelihood` has only a stand-in gradient, so it cannot drive HMC |
| `CSvgp`, `CSviBayesGplvm`, `CDeepGp` | mini-batch ELBO | No (noisy) | Refused: noise breaks the accept step |

The targets are generalised posteriors (bound, or objective at fixed sites, times prior), not posteriors over the variational parameters or sites. The bound models do not add kernel priors to their objective, so the adaptor adds them (value and gradient) and a prior on the variance 1/beta.

## Acceptance Criteria

- [x] Finite-difference gradient (priors and Jacobian on) for `CBayesGplvm`, `CMrd`, `CPpa`
- [x] Prior terms equal their closed form; improper targets refused
- [x] q(X), Z unchanged by sampling; parameters restored in Python
- [x] One-parameter grid comparison for `CBayesGplvm`
- [x] Python `set_prior` and `hmc` for `BayesianGPLVM` and `MRD`; clear refusal for mini-batch models
- [ ] CLI (`bgplvm`) sampling mode: not done; Python and C++ only

## Related

- CIP: 003B
- Tests: `testVariationalHmc`, `python/tests/test_hmc.py`
