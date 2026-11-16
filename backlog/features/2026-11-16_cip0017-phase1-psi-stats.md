---
id: "2026-11-16_cip0017-phase1-psi-stats"
title: "Phase 1: CKern psi statistics (RBF/ARD/Lin/Bias/White/Cmpnd)"
status: "Completed"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "features"
related_cips: ["0017"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- psi
- gplvm
- kernels
---

# Task: Phase 1 psi statistics on CKern

> Backlog tasks execute CIPs (HOW). Linked CIP: [0017](../../cip/cip0017.md).

## Description

Add virtual psi0 / psi1 / psi2 (and getGradPsi) to `CKern`, implement closed forms for RBF, RBF-ARD, Linear, Bias, White, and additive compounds (with psi2 cross terms), and verify with Monte Carlo and finite differences in `testPsi`.

## Acceptance Criteria

- [x] Default `CKern` psi methods throw `NotImplementedError`
- [x] RBF / RBFARD / Lin / Bias / White forward psi match MC within tol
- [x] Leaf-kernel `getGradPsi` matches finite differences
- [x] `CCmpndKern` psi0/psi1 additive; psi2 includes cross terms (GPy `Add` rules)
- [x] `make testPsi` is wired and green on macOS OpenBLAS build

## Implementation Notes

Follow GPy `kern.src.psi_comp.rbf_psi_comp` / `linear_psi_comp` and `kern.src.add` cross-term rules. Map GPc RBF-ARD (`inverseWidth` × `scales`) to per-dimension lengthscales. Avoid allocating full \(N\times M\times M\) in the hot path; accumulate \(\Psi_2\) into \(M\times M\).

## Related

- CIP: 0017
- Requirement: REQ-000B

## Progress Updates

### 2026-11-16

Task created; CIP-0017 Accepted → In Progress. Implementation started on `cip0017-bayesian-gplvm` worktree.

### 2026-11-16 (complete)

Phase 1 landed: `psi0`/`psi1`/`psi2`/`getGradPsi` on `CKern` with RBF, RBFARD, Lin, Bias, White, Cmpnd. `./testPsi` reports 0 failures (MC + FD).
