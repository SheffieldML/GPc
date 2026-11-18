---
id: "2026-11-18_cip0021-phase3-testppa"
title: "CIP-0021 Phase 3: testPpa vs Phase 0 fixtures + CLI smoke"
status: "Proposed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["0021"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0021-phase2-em-port"]
tags:
- backlog
- ppa
- testing
- verification
---

# Task: CIP-0021 Phase 3 — verification

## Description

Add `testPpa` comparing C++ init / short-EM state to `testdata/ppa/phase0_*` ASCII fixtures; `checkGradients` on kernel objective; CLI learn smoke on the N=30 subset.

## Acceptance Criteria

- [ ] `testPpa` builds and passes against `testdata/ppa/`
- [ ] Kernel `checkGradients` passes on a small problem after init
- [ ] `./ppa learn` smoke on fixture data (or documented subset path)
- [ ] Existing `tests` / `testIvm` remain green

## Related

- CIP: 0021
- Fixtures: `testdata/ppa/`

## Progress Updates

### 2026-11-18
Task created after Phase 0.
