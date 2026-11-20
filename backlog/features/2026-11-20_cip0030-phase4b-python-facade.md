---
id: "2026-11-20_cip0030-phase4b-python-facade"
title: "CIP-0030 follow-on: Python façade for CSviBayesGplvm"
status: "Ready"
priority: "Medium"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0030-phase4-surfaces"]
tags:
- backlog
- python
- svi
- bayesian-gplvm
---

# Task: CIP-0030 Phase 4b — Python façade for SVI Bayesian GPLVM

## Description

Expose `CSviBayesGplvm` through the CIP-001D `gpc` façade (`wrap_svi_bgplvm` + `gpc.models.SviBayesianGPLVM`), matching the existing GP / SparseGP bind patterns.

## Acceptance Criteria

- [ ] pybind wrapper constructs / optimises / reads \(\boldsymbol\mu,\mathbf S\)
- [ ] `gpc.models` constructor with inducing family and sample/batch kwargs
- [ ] Pytest smoke on a tiny synthetic \(Y\)
- [ ] Documented in `python/README.md`

## Related

- CIP: 0030, 001D

## Progress Updates

### 2026-11-20

Deferred from Phase 4 CLI/stream landing; CLI is the first-class surface for now.
