---
id: "2026-11-22_csvgp-kl-grad-from-chol"
title: "CSvgp: KL gradient from stored Cholesky without pdinv"
status: "Proposed"
priority: "Medium"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["0035", "002A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- svgp
- cholesky
---

# Task: KL gradient from stored Cholesky (no per-output `pdinv`)

## Description

[CIP-0035](../../cip/cip0035.md) Phase 0a specifies that in Cholesky mode, KL and its gradient are \(O(M^2)\) per output. The KL *value* already uses the stored factor via `upperCholOfS` (#141). The KL *gradient* in `klQ` / `qParamsGrad` still forms \(S^{-1}\) with `pdinv(cholS)` — one \(M^3\) triangular invert per output per step — then packs Euclidean \((\partial/\partial m,\partial/\partial S)\) into \((\tilde{\mathbf m},L)\).

Compute \(\partial\mathrm{KL}/\partial L\) (and \(\partial\mathrm{KL}/\partial\tilde{\mathbf m}\)) directly from the stored lower factor with triangular solves / rank-1 updates, avoiding `pdinv`.

## Acceptance Criteria

- [ ] Cholesky-mode `klQ(..., gQ)` / `qParamsGrad` do not call `pdinv` on \(\tilde S_j\)
- [ ] Cost is \(O(M^2)\) (or \(O(M^2)\) triangular work) per output, not a dense invert
- [ ] FD agreement on packed \(q\) params unchanged (`testSvgp`)
- [ ] DeepGP / SVGP suites green

## Implementation Notes

- \(\mathrm{KL}=\frac12(\mathrm{tr}S+\|m\|^2-M-\log|S|)\); with \(S=LL^\top\), \(\partial/\partial L\) is available from \(L^{-1}\) triangular products without forming \(S^{-1}\) densely if preferred, or via a single `trsm` chain.
- Unmeasured today; expected win scales with \(D_\ell M^3\) on wide hidden layers.

## Related

- CIP: 0035 Phase 0a design claim; 002A SVGP
- PR context: out of scope for #142

## Progress Updates

### 2026-11-22

Task created from DeepGP timing follow-ups (alongside MLP bulk grads).
