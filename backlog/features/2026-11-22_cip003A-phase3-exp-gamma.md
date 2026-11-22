---
id: "2026-11-22_cip003A-phase3-exp-gamma"
title: "CIP-003A Phase 3: Exponential and Gamma CNoise"
status: "Completed"
priority: "Medium"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["003A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-22_cip003A-phase2-student-poisson"]
tags:
- backlog
- likelihood
- exponential
- gamma
---

# Task: CIP-003A Phase 3 — Exponential / Gamma

## Description

Add `CExponentialNoise` and `CGammaNoise` with stream registration and verification fixtures.

## Acceptance Criteria

- [x] Stream types `exponential` and `gamma`
- [x] Density / gradient tests green

## Related

- CIP: 003A

## Progress Updates

### 2026-11-22
Completed with Phase 2 test binary extended; Gamma trigamma local helper.
