---
id: "2026-11-21_cip0038-phase-a-wrap"
title: "CIP-0038 Phase A: wrap existing C++ kernels on gpc.kern"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0038"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- python
- facade
- kernels
---

# Task: CIP-0038 Phase A — wrap existing catalogue

## Description

Expose Matérn 3/2 & 5/2, Exponential (OU alias), RatQuad, StdPeriodic, Cosine, Brownian, Linear, Poly (and ARD where `*ard` classes exist) on `gpc.kern` and in `addPartsCompound`, using CIP-0014 GPy path segments.

## Acceptance Criteria

- [x] `gpc.kern` constructors for the Phase A table in CIP-0038
- [x] `addPartsCompound` builds matching `C*Kern` and sets GPy path segments
- [x] `ARD=True` maps to `*ard` stream types where available
- [x] Pytest construction + path round-trips; Brownian rejects `input_dim!=1`
- [x] Help/docstring smoke list updated

## Related

- CIP: 0038
