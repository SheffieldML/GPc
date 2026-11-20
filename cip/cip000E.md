---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000E"
last_updated: "2026-11-20"
status: "Closed"
compressed: true
related_requirements: ["0002"]
related_cips: ["0005", "0012"]
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
- [x] Closed - Verified and complete (superseded-complete by [CIP-0012](cip0012.md))
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Introduced `SparseLinalgStableFactor` (trsm against upper `LcholK` for \(K_{uu}^{-1}K_{uf}\)) and the Legacy dual-path flag. [CIP-0012](cip0012.md) upgrades StableFactor in place to the whitened \(C=I+B\Lambda^{-1}B^\top\) forward path and reverse-mode adjoint gradients. This CIP is **Closed** as superseded-complete: the flag and `diagDtcvar` dual-path harness remain; numerical meaning of StableFactor is owned by 0012.

## Motivation

Sparse approximations repeatedly need \(K_{uu}^{-1} K_{uf}\). Forming \(K_{uu}^{-1}\) with `pdinv` after Cholesky is convenient but numerically weaker than triangular solves against the factor.

## Detailed Description

### Protocol (as shipped; 0012 extends StableFactor)

- `CGp::SparseLinalgLegacy` (default) — `gemm(invK_uu, K_uf)`.
- `CGp::SparseLinalgStableFactor` — after 0012: whitened \(B,C\) path (see 0012); before 0012: two `trsm` for \(V\) only.
- Accessors: `setStableSparseLinalg` / `setSparseLinalgMode`; CLI `gp … -L legacy|stable`.
- Not persisted in `.model` streams.

### Dual-path harness

`diagDtcvar` runs Legacy vs StableFactor LL and analytic gradient max-abs diffs; also FD-checks StableFactor. Extended by 0012’s `diagSparseStable` and `scripts/sparse_mp_reference.py`.

## Implementation Status

- [x] Inventory `invK_uu` consumers for DTC / DTCVAR
- [x] Add `sparseLinalgMode` flag (default Legacy) + optional CLI
- [x] StableFactor forward path for spherical DTC / DTCVAR (and FITC `V`)
- [x] Dual-path diagnostic harness (`diagDtcvar`)
- [x] Documented test / tolerance gate (`docs/numerics.md`)
- [x] CI green; CIP Closed (superseded-complete by 0012, 2026-11-16)

## Testing Strategy

- `make diagDtcvar && ./diagDtcvar` — dual-path + FD under both modes.
- `make USE_MATIO=1 testGp && ./testGp` remains green with default Legacy.

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md)

## Related

- [CIP-0012](cip0012.md) — whitened StableFactor + reverse-mode adjoints (owns current StableFactor meaning)
