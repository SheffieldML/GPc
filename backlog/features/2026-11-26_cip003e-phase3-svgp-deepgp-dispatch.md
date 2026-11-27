---
id: "2026-11-26_cip003e-phase3-svgp-deepgp-dispatch"
title: "Phase 3: SVGP and deep-GP dispatch through the block oracle"
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

# Task: Phase 3: SVGP and deep-GP dispatch through the block oracle

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

CSvgp data term, CVI natgrad sites and elboGrad loop over blocks; CDeepGp final layer wiring.

## Acceptance Criteria

- [x] testSvgp, testDeepGp, testSviBayesGplvm unchanged at blockSize 1
- [x] checkGradients passes with a block likelihood

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.

### 2026-11-27
Completed. `CSvgp::expectedLogLikBatch` evaluates E_q[log p] and the per-latent gradients for a whole batch; scalar-block likelihoods keep the per-element path in its original order, blockSize > 1 uses `expectedLogLikelihoodBlock` (the latents of a datum are independent under q). The CVI natural-gradient step and every `elboGrad*`/`dataTerm*` path read the precomputed per-site gradients, so the CVI site algebra is unchanged. `setLikelihood` requires `CapExpectedLogLik` and a block size that tiles the outputs. `CDeepGp` forwards to its final layer, so it needs no change. Tests: new `testSvgpBlock` (factorised block equals scalar, coupled 2x2-precision Gaussian block gradient matches FD to ~1e-8 for full batch and minibatch, CVI ascent, refusal) and a coupled-block final-layer FD check in `testDeepGp` (~4e-8). `testSvgp`, `testDeepGp`, `testSviBayesGplvm` unchanged. `CSviBayesGplvm` inherits the path through `CSvgp`.
