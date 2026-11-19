---
id: "2026-11-19_cip002D-phase4-qw8-gpu-req"
title: "CIP-002D Phase 4: QW8 timing artifacts + GPU decision REQ"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002D-phase3-qw6-qw7"]
tags:
- backlog
- performance
- bench
- ci
- gpu
---

# Task: CIP-002D Phase 4 (QW8 + GPU decision record)

## Description

1. **QW8** — Extend `benchGPc` with `--kern mlp` and checked-build help text; add fixed-case runner + compare script; CI workflow uploads RELEASE timings and warn-only self-baseline vs main.
2. **GPU decision** — Open [REQ-0016](../../requirements/req0016_gpu-training-decision.md) (Proposed) from CIP-002D Part B; no GPU code in this task.

## Acceptance Criteria

- [x] `benchGPc --kern rbf|mlp` prints `kern=` on timing lines
- [x] Fixed cases cover E1 / E2 sizes / E5 (+ MLP FTC smoke) via `bench/timing_cases.json`
- [x] `.github/workflows/bench-timing.yml` builds `RELEASE=1 bench`, uploads artifacts, warn-only PR compare
- [x] REQ-0016 Proposed with Part B options and acceptance criteria
- [x] CIP-002D Phase 4 checked

## Related

- CIP: 002D
- REQ: 0016

## Progress Updates

### 2026-11-19

Task opened and completed with Phase 4 landing on `cip002D-cpu-quick-wins`.
