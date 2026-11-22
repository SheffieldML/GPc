---
id: "2026-11-22_cip003A-phase4-python-facade"
title: "CIP-003A Phase 4: gpc.likelihoods façade and model wiring"
status: "Completed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["003A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-22_cip003A-phase1-skew-noise"]
tags:
- backlog
- python
- facade
- likelihood
---

# Task: CIP-003A Phase 4 — Python façade

## Description

Expose `gpc.likelihoods` constructors (Gaussian, Bernoulli, Skew*, StudentT, Poisson, … as implemented) and wire `likelihood=` / `inference=` on regression-style models. Update subset checklist and help docstrings.

## Acceptance Criteria

- [x] `import gpc.likelihoods` with GPy-familiar names
- [x] Density smoke tests in `python/tests/test_likelihoods.py`
- [x] Subset checklist updated (Phase 5)
- [x] DeepGP accepts new likelihood type strings

## Related

- CIP: 003A

## Progress Updates

### 2026-11-22
`gpc.likelihoods` + `_gpc.noise_log_density`; DeepGP likelihood strings extended.
