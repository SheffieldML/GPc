---
id: "2026-11-21_cip0031-phase-c-deprecate-facade"
title: "CIP-0031 Phase C: deprecate posteriorMeanVar + Python façade"
status: "Ready"
priority: "Medium"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0031", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0031-phase-b-models"]
tags:
- backlog
- prediction
- python
- deprecation
---

# Task: CIP-0031 Phase C — Deprecate + Python

## Description

Mark `posteriorMeanVar` deprecated; migrate internal callers to latent/predictive; expose `predict` / `predict_noiseless` on the `gpc` façade; update checklist / side-by-side.

## Acceptance Criteria

- [ ] `posteriorMeanVar` documented deprecated; internal callers migrated where practical
- [ ] Python `predict` → predictive; `predict_noiseless` → latent
- [ ] Pytest: predictive variance ≈ noiseless + `likelihood.variance`
- [ ] `subset_checklist.md` / side-by-side updated

## Related

- CIP: 0031, 001D
- Depends on: Phase B

## Progress Updates

### 2026-11-21

Task created Ready on CIP Accept.
