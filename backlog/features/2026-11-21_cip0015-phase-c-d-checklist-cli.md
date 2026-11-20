---
id: "2026-11-21_cip0015-phase-c-d-checklist-cli"
title: "Phase C/D: GPy checklist + CLI -k polish (CIP-0015)"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0015", "000C"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0015-phase-b-demos"]
tags:
- backlog
- tutorials
- cli
- kernels
- gpy
- features
---

# Task: CIP-0015 Phase C + D — checklist and CLI polish

## Description

Finish [CIP-0015](../../cip/cip0015.md) Phase C (complete GPy↔GPc checklist in `docs/tutorials/README.md`) and Phase D (wire CIP-000C kernel types into `gp learn -k` / help). Validate [REQ-0009](../../requirements/req0009_gpy-tutorial-parity-native.md).

## Acceptance Criteria

- [x] Checklist maps every in-scope GPy item to a tutorial/demo or explicit non-goal
- [x] `gp learn -k` constructs matern32/52 (+ard), ratquadard, stdperiodic, cosine, brownian, ibrownian, rbfard
- [x] Help text lists learn-path types; multi-output stays API-demo only; FITC not tutorial default
- [x] Smoke: short `gp learn` with matern32 / stdperiodic / matern52+lin
- [x] REQ-0009 acceptance criteria ticked; CIP-0015 Implemented/Closed

## Related

- CIP: 0015
- REQ: 0009 (primary), 0006 end-to-end path

## Progress Updates

### 2026-11-21

Delivered with docs + `gp.cpp` learn-path wiring. Status → Completed.
