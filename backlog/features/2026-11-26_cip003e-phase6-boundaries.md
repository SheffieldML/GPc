---
id: "2026-11-26_cip003e-phase6-boundaries"
title: "Phase 6: loaders, Python, factory and CLI for multi-class"
status: "Completed"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-27"
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

- [x] Integer labels convert to N x K one-hot at the boundary only (`ndlnoise::oneHotLabels`)
- [x] Python RobustMax/Softmax and DeepGP(likelihood=...)
- [x] CLI (`deepgp learn --likelihood`) and .model round-trip (deepgp stream); `gp`/`ivm` CLIs are out of scope

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.

### 2026-11-27
Completed. Details in the CIP ("Phase 6 results"). `gp` and `ivm` command lines were not extended.
