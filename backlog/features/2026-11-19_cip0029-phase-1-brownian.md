---
id: "2026-11-19_cip0029-phase-1-brownian"
title: "CIP-0029 Phase 1: Brownian erf K_fu closed forms + gates"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0029"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0029-phase-0-skeleton"]
tags:
- backlog
- sparse
- inducing
- greens
- brownian
---

# Task: CIP-0029 Phase 1 — Brownian erf path

## Description

Implement closed-form \(K_{fu}=(g*\psi_m)\) for `CBrownianKern` with Gaussian windows: scaled \(\Phi((x-z_m)/s)\) (erf) units, plus gradients w.r.t. variance, centres \(z_m\), window width \(s\), and data \(x\). Wire through `CInducingGreens` so DTC/DTCVAR sparse paths run end-to-end for 1-D Brownian.

## Acceptance Criteria

- [x] Closed-form greens cross-compute + grads on `CBrownianKern` (API parallel to `wnCross*`)
- [x] Numeric/MC convolution matches closed form on a fine 1-D grid
- [x] FD agreement on kernel params, \(Z\) (X/s covered by analytic forms; variance+Z FD in `testGreensBrownianPhase1`)
- [x] `testCGp` DTCVAR smoke with greens + Brownian
- [x] \(\kappa(K_{uu})\) independent of Brownian variance on a small sweep cell (Phase 4 `greens_var_sweep_*`)

## Implementation Notes

Pin constants against known Brownian covariance and direct convolution. Do not change point-inducing Brownian semantics.

## Related

- CIP: 0029
- Depends: 2026-11-19_cip0029-phase-0-skeleton

## Progress Updates

### 2026-11-19
Task created (Ready) after CIP-0029 Accepted.

### 2026-11-19 (done)
`greensCross*` on Brownian (Wiener ∫₀ᵗ ψ); `CInducingGreens` wired; `testGreensBrownianPhase1` green (quad maxRel ~3e-8).
