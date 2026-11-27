---
id: "2026-11-27_interchange-latent-tests"
title: "Verify GPc against the gp-scuderia GP-LVM fixtures"
status: "Proposed"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0016", "0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-27_interchange-latent-bound-psi"]
tags:
- backlog
- interchange
- gplvm
- gp-scuderia
---

# Task: Verify GPc against the gp-scuderia GP-LVM fixtures

> Driven by gp-scuderia CIP-0006 (GP-LVM comparison tier). The wire format is defined there:
> `models.bayesian_gplvm` with a `latent` block and no `data.X`; the schema, fixtures and the
> GPy reference values live in `gp-scuderia`. GPc implements the reader and two operations.

## Description

Add tests following the project tenet (assert and verify against a reference): GPc reads the gp-scuderia GP-LVM goldens and agrees with the GPy reference values for `bound_without_kl`, `kl` and the psi statistics. Include the oracle-free checks that need no GPy.

## Acceptance Criteria

- [ ] With S close to 0, `bound_without_kl` equals the log likelihood of the equivalent `dtcvar` sparse regression at inputs mu
- [ ] psi1 and psi2 agree with Monte Carlo expectations of the kernel under q(X) within sampling error
- [ ] Agreement with `gp-scuderia/examples/bgplvm_rbf_ard.yaml` reference values at the tolerances recorded there
- [ ] Gradient: `checkGradients` passes for the latent parameters on the fixture

## Implementation Notes

The golden file is produced in gp-scuderia (CIP-0006 step 3); until it lands, test against the first two checks. gp-scuderia's `scripts/check_latent_ops.py` is the model for the oracle-free checks.

## Related

- CIP: 0016 (interchange), 0017 (psi statistics and Bayesian GP-LVM)
- gp-scuderia: `cip/cip0006.md`, backlog `2026-11-27_gplvm-runner-ops`

## Progress Updates

### 2026-11-27
Task created with Proposed status.
