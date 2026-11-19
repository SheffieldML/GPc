---
id: "2026-11-19_cip0028-phase-1-mlp-isotropic"
title: "CIP-0028 Phase 1: MLP isotropic weight-space K_fu closed forms"
status: "Proposed"
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

- [ ] Closed-form `wsCross*` + grads on `CMlpKern` using \(\boldsymbol{\mu}_m,S_m,a_m\) (not \(\hat{\mathbf{w}}_m,\tilde S\) alone)
- [ ] MC of \(\int\phi\sqrt{p}\,\psi\) matches closed form on 1-D and small \(d\); control that \(\int\phi\psi\) alone does *not* match
- [ ] FD agreement on kernel params (incl. weight/bias variances), \(W\), and \(s\) / \(X\) as scoped
- [ ] `testCGp` DTCVAR smoke with weight-space + MLP
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
