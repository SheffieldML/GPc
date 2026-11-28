---
id: "2026-11-27_interchange-latent-bound-psi"
title: "Expose bound and psi statistics through the interchange"
status: "Ready"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0016", "0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_interchange-latent-read"]
tags:
- backlog
- interchange
- gplvm
- gp-scuderia
---

# Task: Expose bound and psi statistics through the interchange

> Driven by gp-scuderia CIP-0006 (GP-LVM comparison tier). The wire format is defined there:
> `models.bayesian_gplvm` with a `latent` block and no `data.X`; the schema, fixtures and the
> GPy reference values live in `gp-scuderia`. GPc implements the reader and two operations.

## Description

Add `CInterchange::bound()` returning `{bound_without_kl, kl}` and `CInterchange::psi()` returning psi0 (n), psi1 (n x m) and psi2 (m x m, summed over n). `varDtcBound()` is already the bound without KL; compute the KL against N(0, I) from mu and S. Bind both in `python/gpc` and add `bound(model)` and `psi(model)` to `gpc.interchange`; gp-scuderia's `gpc_adapter` calls exactly those names.

## Acceptance Criteria

- [ ] `gpc.interchange.bound(model)` returns a dict with float `bound_without_kl` and `kl`
- [ ] `gpc.interchange.psi(model)` returns arrays psi0, psi1, psi2 of the shapes above
- [ ] KL equals 0.5 * sum(mu^2 + S - log S - 1)
- [ ] A model without latent variables raises `InterchangeError`

## Implementation Notes

The psi statistics are cached as members of `CBayesGplvm` (`psi0Sum`, and the psi1/psi2 matrices) behind the bound computation; add const accessors rather than recomputing. psi0 is needed per point, so check whether only the sum is cached. Do not apply `includeKl` here: the contract reports the two terms separately so that frameworks that fold the KL in can be compared.

## Related

- CIP: 0016 (interchange), 0017 (psi statistics and Bayesian GP-LVM)
- gp-scuderia: `cip/cip0006.md`, backlog `2026-11-27_gplvm-runner-ops`

## Progress Updates

### 2026-11-27
Task created with Ready status.
