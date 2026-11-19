---
id: "2026-11-19_cip0028-phase-1-mlp-isotropic"
title: "CIP-0028 Phase 1: MLP isotropic weight-space K_fu closed forms"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0028-phase-0-skeleton"]
tags:
- backlog
- sparse
- inducing
- weight-space
- mlp
---

# Task: CIP-0028 Phase 1 — MLP isotropic weight-space

## Description

Implement closed-form \(K_{fu}\) for `CMlpKern` with Gaussian windows on **unit white noise in weight space** (erf / Williams MLP), including the \(\sqrt{p(\mathbf{w})}\) factor so `weightVariance` / `biasVariance` enter via the Gaussian product \((\boldsymbol{\mu}_m,S_m,a_m)\). Gradients w.r.t. kernel params, weight centres \(W\), window width \(s\), and data \(X\). Wire through `CInducingWeightSpace` so DTC/DTCVAR sparse paths run end-to-end for MLP + weight-space.

## Acceptance Criteria

- [x] Closed-form `wsCross*` + grads on `CMlpKern` using \(\boldsymbol{\mu}_m,S_m,a_m\) (not \(\hat{\mathbf{w}}_m,\tilde S\) alone)
- [x] MC of \(\int\phi\sqrt{p}\,\psi\) matches closed form on 1-D and small \(d\); control that \(\int\phi\psi\) alone does *not* match
- [x] FD agreement on kernel params (incl. weight/bias variances), \(W\), and \(s\) / \(X\) as scoped
- [x] `testCGp` DTCVAR smoke with weight-space + MLP
- [ ] Informative `ws_wv_sweep_*` seed: sparse bound vs exact/dense gap vs `weightVariance` with fixed windows (full cell can land in Phase 3)

## Implementation Notes

Authoritative math: CIP-0028 “Mathematics (MLP / erf)” after 2026-11-19 correction. Pin overall scale against `CMlpKern::computeElement`. Implement the \(\sqrt{p}\) product form for learnable \(w,b\). The omit-\(\sqrt{p}\) / simple-erf formula is the \(w,b\to\infty\) limit — expose via `CArcCosineKern` order 0 in Phase 2 (optional large-\((w,b)\) bridge here). Do not change point-inducing MLP semantics.

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-0-skeleton

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted; blocked on Phase 0.

### 2026-11-19 (math fix)
CIP-0028 corrected: MC oracle is \(\int\phi\sqrt{p}\,\psi\); \(\kappa(K_{uu})\) demoted to sanity. Phase 0 complete — this task is unblocked.

### 2026-11-19 (done)
`CMlpKern` ws\* closed form with √p Gaussian product; FD grads; White zeros + Cmpnd sum; `testWeightSpaceInducingPhase1` green (MC √p max rel ~5e-3; no-√p control fails; DTCVAR SCG smoke). Informative `ws_wv_sweep_*` deferred to Phase 3.

### 2026-11-19 (superseded as shipping design)
NumPy prototype: isotropic overlap collapses with \(d\) (e.g. ~2% variance at \(d=16\), \(M=64\) under \(\kappa\le10^4\)). Shipping design moves to ridge windows — see [`2026-11-19_cip0028-phase-1-ridge`](./2026-11-19_cip0028-phase-1-ridge.md). Keep this isotropic path as a low-\(d\) oracle until ridge lands.
