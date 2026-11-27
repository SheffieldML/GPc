---
id: "2026-11-27_cip003f-phase2-batched-input-grads"
title: "Phase 2: batched inducingGradX hook and kernel overrides (A2)"
status: "Ready"
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

# Task: Phase 2: batched inducingGradX hook and kernel overrides (A2)

> Implements CIP-003F (HOW). See the CIP for the design and gates.

## Description

RBF, RBF-ARD, linear, MLP, MLP-ARD overrides; CMlpKern::inducingGradZ.

## Acceptance Criteria

- [ ] Gates listed for this phase in CIP-003F pass

## Implementation Notes

Work on branch `cip003F-deepgp-training-speed` (worktree `../GPc-cip003F`).

## Related

- CIP: 003F

## Progress Updates

### 2026-11-27
Task created with Ready status when CIP-003F was accepted.
