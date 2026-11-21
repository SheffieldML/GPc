---
id: "2026-11-21_cip0035-python-facade-deepgp"
title: "Python façade for CDeepGp (gpc.models.DeepGP)"
status: "Proposed"
priority: "Low"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0035", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0035-phases2-4"]
tags:
- backlog
- deep-gp
- python
- facade
---

# Task: Python façade for `CDeepGp`

## Description

Expose [CIP-0035](../../cip/cip0035.md) `CDeepGp` through the `gpc` Python package as `gpc.models.DeepGP`, mirroring the existing SVGP / Bayesian GPLVM façades ([CIP-001D](../../cip/cip001D.md)). C++ CLI (`./deepgp`) and `testDeepGp` already cover Phase 4 surfaces.

## Acceptance Criteria

- [ ] `gpc.models.DeepGP` constructs from numpy `X`, `Y`, layer widths, and inducing counts
- [ ] Adam training and ELBO accessible from Python
- [ ] Optional probit / ordered final likelihood
- [ ] Round-trip `.model` load/save via the `deepgp` stream type
- [ ] Short notebook or doctest using the composite / sinc toy

## Related

- CIP: 0035, 001D
- Tutorial: [docs/tutorials/deepgp.md](../../docs/tutorials/deepgp.md)

## Progress Updates

### 2026-11-21
Task created (Proposed) after CIP-0035 Phase 4 landed the C++ surfaces.
