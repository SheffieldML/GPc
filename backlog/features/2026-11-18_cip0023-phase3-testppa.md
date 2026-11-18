---
id: "2026-11-18_cip0023-phase3-testppa"
title: "CIP-0023 Phase 3: testPpa vs Phase 0 fixtures + CLI smoke"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["0023"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0023-phase2-em-port"]
tags:
- backlog
- ppa
- testing
- verification
---

# Task: CIP-0023 Phase 3 — verification

## Description

Add `testPpa` comparing C++ init / short-EM state to `testdata/ppa/phase0_*` ASCII fixtures; `checkGradients` on kernel objective; CLI learn smoke on the N=30 subset.

## Acceptance Criteria

- [x] `testPpa` builds and passes against `testdata/ppa/`
- [x] Kernel LL gradient matches Octave `ppaKernelLogLikeGrad` fixture
- [x] `./ppa learn` smoke on `testdata/ppa/phase0.svml`
- [ ] Existing `tests` / `testIvm` remain green (not re-run in this task)

## Related

- CIP: 0023
- Fixtures: `testdata/ppa/`

## Progress Updates

### 2026-11-18
`testPpa` 0 failures; CLI learn writes `phase0.model`.
