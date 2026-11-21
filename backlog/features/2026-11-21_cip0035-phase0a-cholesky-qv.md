---
id: "2026-11-21_cip0035-phase0a-cholesky-qv"
title: "CIP-0035 Phase 0a: Cholesky-mode q(v) in CSvgp"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0035", "0030", "002A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- deep-gp
- svgp
- cholesky
- adam
---

# Task: CIP-0035 Phase 0a — Cholesky-mode \(q(\mathbf v)\)

> Backlog for [CIP-0035](../../cip/cip0035.md) Phase 0a. Worktree: `GPc-cip0035` / `cip0035-deep-gp`.

## Description

Add a parameterisation \(\tilde S_j=L_j L_j^\top\) (lower triangular, log diagonal) with \((\tilde{\mathbf m}_j,L_j)\) packed into the stochastic parameters so Adam can update \(q(\mathbf v)\). CVI natural-gradient steps remain the default for the final layer; Cholesky mode is additive. KL and its gradient are \(O(M^2)\) per output in this form.

## Acceptance Criteria

- [x] Cholesky-mode \(q(\mathbf v)\) on `CSvgp` (opt-in; CVI default unchanged)
- [x] Adam on \((\tilde{\mathbf m},L)\) reaches the CVI optimum on `testSvgp` problems
- [x] Finite differences on \(q\) parameters and KL gradient pass
- [x] Existing CVI / natgrad paths still green

## Implementation Notes

Shared with the layer API (Phase 0b): `gQ` packing for Adam. Keep CVI storage of \(\tilde S_j\) for the hybrid final-layer scheme.

## Related

- CIP: 0035

## Progress Updates

### 2026-11-21

Task created In Progress on CIP Accept; implementation starting in `GPc-cip0035`.

### 2026-11-21 (complete)

Phase 0a landed: `QModeCholesky`, pack/unpack, `qParamsGrad` FD-green; Adam-on-q reaches CVI ELBO (`./testSvgp` 0 failures).
