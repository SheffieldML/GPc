---
id: "2026-11-25_cip003b-lengthscale-prior-units"
title: "CIP-003B: priors and samples in GPy lengthscale units"
status: "Proposed"
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

- [ ] A gamma prior on `rbf.lengthscale` is accepted and evaluated on the lengthscale
- [ ] Samples report `rbf.lengthscale`; the Jacobian matches finite differences
- [ ] Grid comparison of the lengthscale posterior in C++ and Python
- [ ] Refusal message and docs updated

## Related

- CIP: 003B
