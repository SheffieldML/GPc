---
id: "2026-11-27_interchange-latent-read"
title: "CInterchange reads models.bayesian_gplvm"
status: "Ready"
priority: "High"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["0016", "0017"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- interchange
- gplvm
- gp-scuderia
---

# Task: CInterchange reads models.bayesian_gplvm

> Driven by gp-scuderia CIP-0006 (GP-LVM comparison tier). The wire format is defined there:
> `models.bayesian_gplvm` with a `latent` block and no `data.X`; the schema, fixtures and the
> GPy reference values live in `gp-scuderia`. GPc implements the reader and two operations.

## Description

Teach `CInterchange::fromJson` the latent model type. A document with `type: models.bayesian_gplvm` carries `latent.{dim, mean, variance, prior}`, `inducing_inputs`, `kernel`, a Gaussian `likelihood` and `data.Y` (no `data.X`, `input_dim` or `inference`). Build a `CBayesGplvm` with the document's own q(X) and Z, set the kernel and noise from the wire parameters, and write the same document back from `toJson`.

## Acceptance Criteria

- [ ] fromJson builds the model from `gp-scuderia/tests/valid/bgplvm_rbf_ard.yaml` (via its JSON twin) and `toJson` returns an equivalent document
- [ ] Refuses with a message naming the field: missing `latent`, `data.X` present, non-positive variance, shape mismatches, `approximation` other than `dtcvar`, non-Gaussian likelihood, prior other than `standard_normal`
- [ ] `models.gplvm` (point estimate, `ftc`) is refused with a clear message until the point-estimate task is done
- [ ] Existing GP documents are unaffected

## Implementation Notes

Keep the mapping logic in C++ (the `CInterchange` principle). `CBayesGplvm` currently owns its latent storage via `initStoreage`; use `setLatents` or add a constructor path that takes mu, S and Z directly, rather than PCA initialisation. The wire `lengthscale` and noise conversions are the ones already used for sparse regression.

## Related

- CIP: 0016 (interchange), 0017 (psi statistics and Bayesian GP-LVM)
- gp-scuderia: `cip/cip0006.md`, backlog `2026-11-27_gplvm-runner-ops`

## Progress Updates

### 2026-11-27
Task created with Ready status.
