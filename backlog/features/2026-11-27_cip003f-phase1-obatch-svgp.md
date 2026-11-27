---
id: "2026-11-27_cip003f-phase1-obatch-svgp"
title: "Phase 1: O(B) SVGP minibatch steps (A1)"
status: "Completed"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["003F"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- deepgp
---

# Task: Phase 1: O(B) SVGP minibatch steps (A1)

> Implements CIP-003F (HOW). See the CIP for the design and gates.

## Description

Route small batches through the explicit-input paths in CSvgp::elboGrad, dataTermQGrads and naturalGradientStep; lazy full-data default.

## Acceptance Criteria

- [ ] Gates listed for this phase in CIP-003F pass

## Implementation Notes

Work on branch `cip003F-deepgp-training-speed` (worktree `../GPc-cip003F`).

## Related

- CIP: 003F

## Progress Updates

### 2026-11-27
Task created with Ready status when CIP-003F was accepted.
