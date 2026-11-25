---
id: "2026-11-25_cip0016-phase-a-gpc-io"
title: "CIP-0016: Phase A: GPc read/write of the shared format"
status: "In Progress"
priority: "High"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["0016"]
owner: "Neil Lawrence"
dependencies: ["2026-11-25_cip0016-phase0-publish"]
tags:
- backlog
- interop
- gp-scuderia
---

# Task: CIP-0016 Phase A: GPc read/write of the shared format

## Description

Pin gp-scuderia; write/read FTC Gaussian regression with compound kernels via the registry and CIP-0014 conversions; load happens off the SCG path; `.model` untouched; negative tests for unknown type, truncated params, dimension mismatch.

## Acceptance Criteria

- [x] A0: Python adapter `gpc.interchange` matches goldens' `x-reference` (done 2026-11-25)
- [ ] A1: C++ `CInterchange` reads and writes the JSON twin; `testInterchange` checks both goldens against `x-reference` and rejects unknown type, wrong dimension, wrong schema version, missing data, truncated parameters
- [ ] A1: no new required dependency (vendored header-only parser; YAML optional); load off the SCG path
- [ ] A2: `gp learn --from model.json`, `gp export`, `gp import`; `.model` untouched
- [ ] A3: Python delegates to A1; `gp-scuderia/scripts/conformance.sh` covers the C++ path

## Implementation Notes

Mapping rules to implement in C++ (reference: `python/gpc/interchange.py`): `lengthscale` to `inverseWidth` (1/l^2); ARD as `rbfard` with `inverseWidth = min(l)^-2` and `inputScale_j = (min l / l_j)^2`; sparse noise via `beta = 1/variance`; FTC with `kern.white` enables white-plus-likelihood noise (`allowWhiteAndLikelihoodNoise`); zero mean.

## Related

- CIP: 0016
- Requirement: 000A

## Progress Updates

### 2026-11-25
Task created with Ready status on acceptance of CIP-0016.

### 2026-11-25 (later)
Status In Progress. A0 done. Scope split into A0 to A3 after deciding C++ is the canonical implementation (CIP-0016 decision 10).
