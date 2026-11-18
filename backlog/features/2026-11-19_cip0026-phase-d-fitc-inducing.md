---
id: "2026-11-19_cip0026-phase-d-fitc-inducing"
title: "CIP-0026 Phase D (optional): FITC/PITC and free inducing"
status: "Proposed"
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

Extend sparse PPA beyond DTC: FITC/PITC approx types, optional free inducing optimisation, and CIP-001C white-noise inducing features with PPA sites.

## Acceptance Criteria

- [ ] FITC and/or PITC selectable on sparse PPA path with smoke tests
- [ ] Optional unfixed inducing with gradient path via CGp
- [ ] Document interaction with white-noise inducing (CIP-001C)

## Implementation Notes

- Depends on a usable sparse monitor (Phase C).
- Keep optional; do not block Phases A–C.

## Related

- CIP: 0026, 001C

## Progress Updates

### 2026-11-19

Task created (Proposed / optional).
