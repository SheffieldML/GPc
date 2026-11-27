---
id: "2026-11-26_cip003e-phase4-robustmax"
title: "Phase 4: CRobustMaxNoise"
status: "Completed"
priority: "High"
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

### 2026-11-27
Completed. `CRobustMaxNoise` (K classes, fixed eps; type `robustmax`, registered in `readNoiseFromStream`) implements `expectedLogLikelihoodBlock`, `cavityMomentsBlock`, `logZBlock`, `predictBlock`, `out`/`likelihoods`/`logLikelihood`, versioned stream I/O and display. No Laplace capability. Tests in `testRobustMax`: K=2 closed form and probit-step Z (1e-10), reference fixture (P to 3e-13), Monte Carlo (1e-3), FD gradients in mu, var, eps (1e-10 to 1e-7), refusals, `.model` round trip, K=3 block EP with dL/dK matching FD (1.18064). Deviations: the fixture is a stdlib reference, not GPflow output; eps is not yet learnable; K=2 equality with probit is checked through Z, not through a full EP run (robust-max K=2 acts on the difference of two latents). Not covered: ADF with the coupled block, SVGP end-to-end, factory/CLI (Phase 6).
