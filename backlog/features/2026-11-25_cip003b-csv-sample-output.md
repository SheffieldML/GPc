---
id: "2026-11-25_cip003b-csv-sample-output"
title: "CIP-003B: CSV output for HMC samples"
status: "Completed"
priority: "High"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["003B"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- hmc
- sampling
- csv
- interoperability
---

# Task: CSV output for HMC samples

## Description

Write `CHmc` samples as dense CSV so they load directly into pandas, NumPy or arviz, in line with the ecosystem-interoperability tenet. Columns are `iteration`, `log_posterior`, then one column per sampled parameter, named as in `CPosteriorTarget::sampledParams` (for example `rbf.lengthscale`, `likelihood.variance`).

## Acceptance Criteria

- [x] `writeSamplesCsv(target, samples, logPost, out, constrained)` in `CHmc`
- [x] Constrained values by default (positive length scales and variances); unconstrained optimiser values on request
- [x] Round-trip test in `testHmc`: parse the CSV and compare to the in-memory samples to full double precision
- [x] `log_posterior` documented as the sampled target density (includes the log-Jacobian when on)

## Implementation Notes

- 17 significant digits so that round trips are exact.
- The header uses the parameter names already used by `fix()` and `setByPath`.

## Related

- CIP: 003B (Phase 5)
- Tenet: ecosystem-interoperability

## Progress Updates

### 2026-11-25
Task created and started.

### 2026-11-25
Implemented and verified: testHmc (CSV round trip, addPriorByPath) passes; end-to-end `gp sample` run on a toy GP gives 3000 samples, acceptance 0.89, positive constrained values. Also fixed unset betaTransform for FTC models read from a stream (gp relearn segfault).

### 2026-11-25
Implemented and verified: testHmc (CSV round trip, addPriorByPath) passes; end-to-end `gp sample` run on a toy GP gives 3000 samples, acceptance 0.89, positive constrained values. Also fixed unset betaTransform for FTC models read from a stream (gp relearn segfault).
