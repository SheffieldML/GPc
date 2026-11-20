---
id: "2026-11-20_cip0030-phase0b-explicit-input-core"
title: "CIP-0030 Phase 0b: CSvgp explicit-input core with input gradients"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "002A", "002D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002A-inducing-interface"]
tags:
- backlog
- svgp
- svi
- bayesian-gplvm
- input-gradients
---

# Task: CIP-0030 Phase 0b — explicit-input core with \(\partial\)ELBO\(/\partial\mathbf X\)

## Description

Factor `CSvgp`'s data term, sites and gradients into methods that take an explicit input block \(\tilde{\mathbf X}_B\), row indices into \(\mathbf Y\), a minibatch weight \(c\), and an optional `CMatrix* gX`. The public API keeps calling them with rows of the model's fixed \(\mathbf X\).

This is the layer surface for `CSviBayesGplvm` (Phase 1) and, later, for stacking deep-GP layers. Do **not** put \(\mathrm{KL}(q(\mathbf X)\|p(\mathbf X))\) or ownership of \((\boldsymbol\mu,\log\mathbf S)\) in the core.

## Acceptance Criteria

- [x] Core methods evaluate the data term / sites / grads at caller-supplied \(\tilde{\mathbf X}_B\) (`dataTermAt`, `elboGradAt`, `naturalGradientStepAt`)
- [x] Optional `gX` filled via `CInducing::gradFromKuuKuf(..., wantX=true)` plus `getDiagGradX`
- [x] Public `CSvgp` behaviour and `testSvgp` unchanged when `gX` is null (cached full-A path retained)
- [x] Finite-difference checks of \(\partial\)ELBO\(/\partial\mathbf X\) for points, white-noise RBF, weight-space ridge, and Green's (1-D)
- [x] Batch-partition exactness at fixed inputs

## Implementation Notes

Depends on Phase 0a (`CSvgp` owns `CInducing*`). Reuse CIP-002D QW6 batched BLAS-3 form. Keep sampling / reparameterisation out of `CSvgp`. Methods are public (owned-core pattern for `CSviBayesGplvm`, not inheritance).

## Related

- CIP: 0030 (owner); 002A (SVGP); 002D (batched form)

## Progress Updates

### 2026-11-20
Task created (Ready) after CIP-0030 Accepted → In Progress.

### 2026-11-20 (later)
Implemented and verified: `testSvgp` Phase 0b checks green (points / WN / weight-space / Green's FD; partition exactness). Status Completed.
