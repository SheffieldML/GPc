---
id: "2026-11-26_cip003e-phase5-softmax-laplace"
title: "Phase 5: CSoftmaxNoise and Laplace block W"
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

# Task: Phase 5: CSoftmaxNoise and Laplace block W

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

Common-random-number sampled expectation and cavity moments; pointDerivsBlock with structured Hessian; block-diagonal W in CLaplaceInference.

## Acceptance Criteria

- [ ] Fixed-seed finite-difference gradients
- [ ] Laplace mode and Hessian vs finite differences
- [ ] Softmax EP with fixed seeds and statistical tolerance

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.
