---
id: "2026-11-22_cip003A-phase4-python-facade"
title: "CIP-003A Phase 4: gpc.likelihoods façade and model wiring"
status: "Ready"
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

- [ ] `import gpc.likelihoods` with GPy-familiar names
- [ ] Tiny seeded Laplace fit smoke tests in `python/tests/`
- [ ] Subset checklist updated

## Related

- CIP: 003A
