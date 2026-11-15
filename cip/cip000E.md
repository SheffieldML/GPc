---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000E"
last_updated: "2026-11-15"
status: "Implemented"
compressed: false
related_requirements: ["0002"]
related_cips: ["0005"]
tags:
- cip
- numerics
- sparse
- dtc
- dtcvar
- fitc
- lapack
title: "Stable sparse-GP factorisation path with dual-path compare flag"
---

# CIP-000E: Stable sparse-GP factorisation path with dual-path compare flag

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add an alternate sparse-GP linear-algebra path that follows the Seeger–Hetzel–Dai–Meissner–Lawrence MXNet pattern: Cholesky of \(K_{uu}\), triangular solve for \(B = L_u^{-1} K_{uf}\), then SYRK to form the inducing-side Gram matrix — avoiding explicit \(K_{uu}^{-1}\) wherever possible. Expose a runtime flag so the new path can be compared against the current `pdinv`/`gemm(invK_uu, …)` originals on likelihood and gradients before any default switch.

## Motivation

Sparse approximations repeatedly need \(K_{uu}^{-1} K_{uf}\). Forming \(K_{uu}^{-1}\) with `pdinv` after Cholesky is convenient but numerically weaker than triangular solves against the factor. Recent DTCVAR diagnostics showed gradients self-consistent — this is hygiene and dual-path verification, not a correctness emergency.

## Detailed Description

### Protocol

- `CGp::SparseLinalgLegacy` (default) — `gemm(invK_uu, K_uf)`.
- `CGp::SparseLinalgStableFactor` — two `trsm` against upper `LcholK` (`U^{-T}` then `U^{-1}`).
- Accessors: `setStableSparseLinalg` / `setSparseLinalgMode`; CLI `gp … -L legacy|stable`.
- Not persisted in `.model` streams.
- `_multiplyInvKuu` used for DTCVAR/FITC forward `V` and DTCVAR gradient `invK_uuK_uf`. Gradients still keep `pdinv` of \(K_{uu}\) available.

### Dual-path harness

`diagDtcvar` runs Legacy vs StableFactor LL and analytic gradient max-abs diffs on the fixed synthetic problem; also FD-checks StableFactor.

## Implementation Status

- [x] Inventory `invK_uu` consumers for DTC / DTCVAR
- [x] Add `sparseLinalgMode` flag (default Legacy) + optional CLI
- [x] StableFactor forward path for spherical DTC / DTCVAR (and FITC `V`)
- [x] Dual-path diagnostic harness (`diagDtcvar`)
- [x] Documented test / tolerance gate (`docs/numerics.md`)
- [ ] CI green; CIP Closed

## Testing Strategy

- `make diagDtcvar && ./diagDtcvar` — dual-path + FD under both modes.
- `make USE_MATIO=1 testGp && ./testGp` remains green with default Legacy.

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md)
