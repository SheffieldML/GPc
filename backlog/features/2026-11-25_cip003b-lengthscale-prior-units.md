---
id: "2026-11-25_cip003b-lengthscale-prior-units"
title: "CIP-003B: priors and samples in GPy lengthscale units"
status: "Completed"
priority: "Medium"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["003B"]
owner: "Neil Lawrence"
dependencies: ["2026-11-25_cip003b-python-hmc"]
tags:
- backlog
- hmc
- priors
- gpy
---

# Task: priors and samples in GPy lengthscale units

## Description

The RBF kernel stores the inverse width gamma natively; the GPy-unit path `rbf.lengthscale` is `1/sqrt(gamma)` (and `1/gamma` for the exponential kernel). Priors and samples currently live on the native value, so they are named `rbf.inv_lengthscale` and a prior on `rbf.lengthscale` is refused (it would otherwise land silently on the wrong quantity). GPy users expect a prior on the lengthscale itself.

Proposed approach: a prior wrapper that evaluates a distribution on the path-unit value (log q(l(gamma)), chain rule in the gradient), and a power transform `x = exp(-c a)` (c = 1/2 for RBF, 1 for exponential) so the sampler reports the lengthscale and applies the matching Jacobian. `CReciprocalExpTransform` is the c = 1 case already used for sparse `likelihood.variance`.

## Acceptance Criteria

- [x] A gamma prior on `rbf.lengthscale` is accepted and evaluated on the lengthscale
- [x] Samples report `rbf.lengthscale`; the Jacobian matches finite differences
- [x] Grid comparison of the lengthscale posterior in C++ and Python
- [x] Refusal message and docs updated

## Related

- CIP: 003B

## Progress Updates

### 2026-11-25
Completed. Design differs from the proposal in one respect: the sampler keeps the stored inverse width and its existing transform, and only the reported value is converted (`SampledParam::lengthscalePower`). A power transform was not needed because the Jacobian for the stored width is unchanged; the change of variables for the prior lives in `CPowerViewDist` (stream type `powerview`, persists with the model). `rbf.inv_lengthscale` remains available for a prior on the stored width. Samples are always reported as lengthscales; `--unconstrained` columns keep the stored-width name. Tests: `testHmc` (prior integrates to one, gradient FD, stream round trip, naming) and `python/tests/test_hmc.py` (grid in lengthscale units for both prior names; exact prior-value check).
