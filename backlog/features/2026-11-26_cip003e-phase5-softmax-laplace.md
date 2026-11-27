---
id: "2026-11-26_cip003e-phase5-softmax-laplace"
title: "Phase 5: CSoftmaxNoise and Laplace block W"
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

# Task: Phase 5: CSoftmaxNoise and Laplace block W

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

Common-random-number sampled expectation and cavity moments; pointDerivsBlock with structured Hessian; block-diagonal W in CLaplaceInference.

## Acceptance Criteria

- [x] Fixed-seed finite-difference gradients
- [x] Laplace mode and Hessian vs finite differences
- [x] Softmax EP with fixed seeds and statistical tolerance

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.

### 2026-11-27
Completed. `CSoftmaxNoise` (type `softmax`, factory and stream registered) with fixed-draw Monte Carlo expectation, log Z, cavity moments and predictions, plus exact `pointDerivsBlock` (g = delta - pi, W = diag(pi) - pi pi^T, dW/df). `CLaplaceInference::runBlock` and `predictiveMeanVarBlock` give a dense block-W Laplace; `CGp` Laplace accepts coupled blocks. Tests: `testSoftmax` (fixed-draw gradients 3e-11, K=2 vs logistic quadrature, point derivatives 1e-10, Laplace stationarity 3e-10, dL/dK for lengthscale and amplitude vs FD 1e-5, blockSize-1 equality with the scalar Laplace 1e-10, block EP with fixed seed recovers labels, stream round trip) and `testCGpSoftmaxLaplace`. Deviations: the Laplace solve is the dense O((N b)^3) form, not Algorithm 3.4; the EP check is a label-recovery smoke test with damping, not an FD gradient check; not covered: ADF with softmax, SVGP end-to-end with softmax, EP dL/dK for the sampled moments.
