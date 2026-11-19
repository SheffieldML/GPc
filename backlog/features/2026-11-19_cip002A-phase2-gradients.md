---
id: "2026-11-19_cip002A-phase2-gradients"
title: "CIP-002A Phase 2: ELBO gradients for kernel, Z, beta via CCholAdjoint"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["002A", "0012"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- svgp
- gradients
- chol-adjoint
---

# Task: CIP-002A Phase 2 — hyperparameter / Z / β gradients

## Description

Extend `CSvgp` so the ELBO (full batch or minibatch, whitened \(q(\mathbf{v})\) held fixed) has analytic gradients w.r.t. kernel parameters, inducing inputs \(\mathbf{Z}\), and Gaussian precision \(\beta\). Route \(\partial\mathcal{L}/\partial\mathbf{A}\) through `trsmAdjointUT` and `cholAdjointUpperBlocked` (CIP-0012), then into `getGradTransParams` / `getDiagGradTransParams` / `getGradX`. Finite-difference check; optional full-batch Adam + natgrad recovers the DTCVAR hyperparameter optimum on a small problem.

## Acceptance Criteria

- [ ] `elboGrad` (or equivalent) fills kernel / Z / β gradients at fixed whitened \(q\)
- [ ] Finite differences agree within a documented tolerance on a small problem
- [ ] Packed `getOptParams` / `setOptParams` / unconstrained gradient with positive transforms for β and kernel
- [ ] Rename `CBayesGplvm` "natural grads" comment to "untransformed grads" to avoid clash with Fisher natural gradients

## Related

- CIP: 002A
- REQ: 0015

## Progress Updates

### 2026-11-19
Task created on CIP acceptance; implementation started on `cip002A-svgp`.

### 2026-11-19
Marked Completed after `testSvgp` verification on `cip002A-svgp`.
