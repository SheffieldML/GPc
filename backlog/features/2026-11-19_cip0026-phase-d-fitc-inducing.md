---
id: "2026-11-19_cip0026-phase-d-fitc-inducing"
title: "CIP-0026 Phase D (optional): FITC/PITC and free inducing"
status: "Completed"
priority: "Low"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0026"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0026-phase-a-csparseppa", "2026-11-19_cip0026-phase-c-dtcvar-bound"]
tags:
- backlog
- ppa
- fitc
- pitc
- inducing
---

# Task: CIP-0026 Phase D — FITC/PITC / inducing

> Optional backlog for [CIP-0026](../../cip/cip0026.md) Phase D.

## Description

FITC/PITC smoke tests and optional free inducing optimisation via `CGp`.

## Acceptance Criteria

- [x] FITC and PITC selectable with finite `ppaBound` smoke tests
- [x] Optional unfixed inducing with short optimise path
- [ ] Document interaction with white-noise inducing (CIP-001C) — deferred note in CIP

## Related

- CIP: 0026, 001C

## Progress Updates

### 2026-11-19

FITC/PITC + free inducing smoke in `testSparsePpa`.
