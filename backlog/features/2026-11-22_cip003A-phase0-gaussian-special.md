---
id: "2026-11-22_cip003A-phase0-gaussian-special"
title: "CIP-003A Phase 0: inverse Mills helpers + skew fixture generators"
status: "Completed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["003A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- likelihood
- skew
- mills
---

# Task: CIP-003A Phase 0 — Gaussian special helpers

## Description

Add shared inverse Mills ratio helpers (`invMillsRatio`, derivatives) in `ndlnoise`, verified against SciPy / GPy `gaussian_special`. Add a small generator under `cip/cip003A/` for skew density fixtures.

## Acceptance Criteria

- [x] `ndlnoise` helpers for λ=φ/Φ and λ′, λ″
- [x] Unit checks vs known values / SciPy-generated references
- [x] Fixture generator script checked in under `cip/cip003A/`

## Related

- CIP: 003A
- REQ: 001B

## Progress Updates

### 2026-11-22
Completed: helpers in `CNoise.h`/`CNoise.cpp`; `testSkewNoise` Mills cases green; `cip/cip003A/gen_skew_fixture.py`.
