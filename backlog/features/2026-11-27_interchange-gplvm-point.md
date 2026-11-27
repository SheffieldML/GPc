---
id: "2026-11-27_interchange-gplvm-point"
title: "Interchange for the point-estimate GP-LVM (models.gplvm)"
status: "Proposed"
priority: "Low"
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

# Task: Interchange for the point-estimate GP-LVM (models.gplvm)

> Driven by gp-scuderia CIP-0006 (GP-LVM comparison tier). The wire format is defined there:
> `models.bayesian_gplvm` with a `latent` block and no `data.X`; the schema, fixtures and the
> GPy reference values live in `gp-scuderia`. GPc implements the reader and two operations.

## Description

`models.gplvm` is `approximation: ftc` with a point estimate of the latents: `latent.mean` only, no variance or prior, no inducing inputs. Map it onto GPc's existing GP-LVM class, and decide what `bound` returns (the marginal likelihood at the point estimate, with `kl` of zero) so that the same operation applies.

## Acceptance Criteria

- [ ] fromJson and toJson round trip for `gp-scuderia/tests/valid/gplvm_point.yaml`
- [ ] `bound` returns the exact log marginal likelihood at the latent point, and `kl` is 0
- [ ] Refuses `latent.variance`, `latent.prior` and `inducing_inputs`

## Implementation Notes

Low priority: CIP-0006 benchmarks one and two use `models.bayesian_gplvm`. Do this when a point-estimate comparison is wanted.

## Related

- CIP: 0016 (interchange), 0017 (psi statistics and Bayesian GP-LVM)
- gp-scuderia: `cip/cip0006.md`, backlog `2026-11-27_gplvm-runner-ops`

## Progress Updates

### 2026-11-27
Task created with Proposed status.
