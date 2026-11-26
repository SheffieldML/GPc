---
id: "2026-11-26_cip003e-phase6-boundaries"
title: "Phase 6: loaders, Python, factory and CLI for multi-class"
status: "Ready"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003E"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- likelihood
- multiclass
---

# Task: Phase 6: loaders, Python, factory and CLI for multi-class

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

One-hot conversion at loaders and Python; gpc.likelihoods.RobustMax/Softmax; DeepGP likelihood option; type-string factory and CClctrl entries.

## Acceptance Criteria

- [ ] Integer labels convert to N x K one-hot at the boundary only
- [ ] Python RobustMax/Softmax and DeepGP(likelihood=...)
- [ ] CLI and .model round-trip

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.
