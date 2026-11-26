---
id: "2026-11-26_cip003e-phase7-experiments"
title: "Phase 7: K-class example and MNIST deep GP versus Gaussian one-hot baseline"
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

# Task: Phase 7: K-class example and MNIST deep GP versus Gaussian one-hot baseline

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

Synthetic K-class example and MNIST deep GP with robust-max, reported against the existing Gaussian one-hot baseline and the MLP/conv baselines.

## Acceptance Criteria

- [ ] Example under examples/
- [ ] MNIST results with accuracy and NLL

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.
