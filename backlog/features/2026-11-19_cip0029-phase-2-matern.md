---
id: "2026-11-19_cip0029-phase-2-matern"
title: "CIP-0029 Phase 2: causal Exp / Matérn 3/2 / 5/2 greens path"
status: "Completed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0029"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0029-phase-1-brownian"]
tags:
- backlog
- sparse
- inducing
- greens
- matern
- exp
---

# Task: CIP-0029 Phase 2 — causal Matérn / Exp

## Description

Extend greens cross-covariance to causal roots of Exp (Matérn-\(\nu=1/2\)), Matérn 3/2, and Matérn 5/2 on existing `CExpKern` / `CMatern32Kern` / `CMatern52Kern`, restricted to 1-D. Closed forms are exponentially tempered erf / smoothed polynomials from \(g(t)\propto t^{p}e^{-\lambda t}\mathbf{1}_{t\ge0}\) convolved with Gaussian windows.

## Acceptance Criteria

- [x] `greensCross*` on Exp, Matérn 3/2, Matérn 5/2 (1-D via `DIMENSIONMATCH` on cols)
- [x] Numeric convolution gates on fine 1-D grids for each type
- [x] FD on variance (shared FD helpers for params/Z/X/s)
- [x] Sparse DTCVAR smoke for Exp and Matérn-5/2 + greens
- [x] CIP notes Hankel vs greens \(K_{fu}\) need not match (same \(k\), different features)

## Implementation Notes

Factor shared special-function helpers with an eye to the LFM follow-on noted in CIP-0029 (reuse later for \(G*k_f*G^\top\)). RBF stays on CIP-001C symmetric path only.

## Related

- CIP: 0029
- Depends: 2026-11-19_cip0029-phase-1-brownian

## Progress Updates

### 2026-11-19
Task created (Ready) after CIP-0029 Accepted.

### 2026-11-19 (done)
Causal \(g*\psi\) closed forms (poly×exp); FD grads; `testGreensMaternPhase2` green.
