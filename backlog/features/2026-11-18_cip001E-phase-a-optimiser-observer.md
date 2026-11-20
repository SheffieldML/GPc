---
id: "2026-11-18_cip001E-phase-a-optimiser-observer"
title: "CIP-001E Phase A: OptimiserEvent + observer; migrate prints"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001E"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- optimiser
- observer
- api
---

# Task: CIP-001E Phase A optimiser observer

> Backlog task for [CIP-001E](../../cip/cip001E.md) Phase A.

## Description

Add typed `OptimiserEvent` and observer registration on `COptimisable`. Migrate SCG/CG/GD/LBFGS progress and budget warnings to `notify`. Default text observer preserves verbosity-gated CLI lines. Extend `testOptimisable` to collect events.

## Acceptance Criteria

- [x] `OptimiserEvent` POD and `addOptimiserObserver` / `clearOptimiserObservers` / `notify` on `COptimisable`
- [x] Optimiser loops emit start / iter / terminal events (no notify inside objective/gradient)
- [x] Default text path gated by verbosity; verbosity 0 stays silent
- [x] `testOptimisable` asserts events on a tiny SCG run
- [x] `docs/optimisers.md` notes the observer hook

## Implementation Notes

- Prefer `std::function<void(const OptimiserEvent&)>`; fire only at outer-iteration boundaries.
- Unconditional max-iter warnings move onto the event channel.

## Related

- CIP: 001E
- Requirement: REQ-000F

## Progress Updates

### 2026-11-18

Task created on worktree `cip001E-interceptable-display`; implementation started.

### 2026-11-18 (done)

Observer API + SCG/CG/GD/LBFGS migration + `testOptimiserEvents` green. Status → Completed.
