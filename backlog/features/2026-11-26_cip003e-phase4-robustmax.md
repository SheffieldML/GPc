---
id: "2026-11-26_cip003e-phase4-robustmax"
title: "Phase 4: CRobustMaxNoise"
status: "Ready"
priority: "High"
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

# Task: Phase 4: CRobustMaxNoise

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

Quadrature expectation and cavity moments, gradients, eps, versioned stream I/O, display, factory entry, predictBlock.

## Acceptance Criteria

- [ ] P(argmax) vs Monte Carlo and high-order quadrature
- [ ] K = 2 equals CProbitNoise expected log-lik, gradients and EP
- [ ] K = 3 GPflow RobustMax fixture under testdata/ with generating script
- [ ] Finite-difference gradients in mu, var, eps
- [ ] .model round-trip at ndlutil::MATCHTOL

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.
