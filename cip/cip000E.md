---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000E"
last_updated: "2026-11-15"
status: "Proposed"
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
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add an alternate sparse-GP linear-algebra path that follows the Seeger–Hetzel–Dai–Meissner–Lawrence MXNet pattern: Cholesky of \(K_{uu}\), triangular solve for \(B = L_u^{-1} K_{uf}\), then SYRK to form the inducing-side Gram matrix — avoiding explicit \(K_{uu}^{-1}\) wherever possible. Expose a runtime flag so the new path can be compared against the current `pdinv`/`gemm(invK_uu, …)` originals on likelihood and gradients before any default switch.

**Which requirements does this CIP address?** REQ-0002 (runnable verification): dual-path compare strengthens the assert-and-verify gate for DTC / DTCVAR (and later FITC).

## Motivation

Sparse approximations (DTC, DTCVAR / VarDTC, FITC, PITC) repeatedly need quantities of the form \(K_{uu}^{-1} K_{uf}\) and \(\hat{K} = K_{uf}^\top K_{uu}^{-1} K_{uf}\). Forming \(K_{uu}^{-1}\) with `pdinv` after Cholesky is convenient but strictly worse numerically than solving triangular systems against the factor.

[Auto-Differentiating Linear Algebra](https://arxiv.org/abs/1710.08717) (Seeger, Hetzel, Dai, Meissner, Lawrence) implemented this as MXNet `linalg` operators (`potrf`, `trsm`, `syrk`) and showed the sparse-GP criterion in that form. GPc already has the LAPACK wrappers (`jitChol`, `trsm`, `syrk`, `gemm`); the gap is that `CGp::_updateInvK` and parts of `updateAD` still materialise `invK_uu` and multiply.

Recent DTCVAR diagnostics (`backlog/bugs/2026-11-15_dtcvar-numerical-diagnostic.md`) found gradients self-consistent and LL matching GPy on a well-conditioned toy problem — so this is **not** a correctness emergency. It is a numerical hygiene and verification improvement: give ourselves a second implementation we can cross-check before trusting either under poor conditioning.

## Detailed Description

### Feasibility

**Easy for the forward sparse path (spherical DTC / DTCVAR).** The MXNet sketch maps almost one-to-one onto existing `CMatrix` calls:

```text
Lu = jitChol(K_uu)              # already in _updateInvK as LcholK
B  = trsm(Lu, K_uf)             # B = Lu^{-1} K_uf  (replace gemm(invK_uu, K_uf))
A  = syrk(B)/β + I   or equiv.  # or keep A = K_uu + K_uf K_uf^T / β (current DTC)
```

Current DTC already builds \(A = K_{uu} + K_{uf} K_{uf}^\top / \beta\) then Choleskys \(A\) — that part is fine. The win is eliminating (or deferring) full `invK_uu.pdinv(LcholK)` for likelihood and for DTCVAR’s diagonal correction \( \mathrm{diag}(K_{ff} - V) \) where \(V\) currently comes from `gemm(invK_uu, K_uf)`.

**Moderate for gradients.** Kernel / inducing gradients today use `invK_uu`, `Ainv`, and related caches (`gK_uu.deepCopy(invK_uu)` etc.). Options:

1. Keep computing `invK_uu` via `pdinv` for gradient-only consumers even on the stable forward path (partial win; dual-path still meaningful for LL).
2. Rewrite gradient expressions to use `trsm` / `trmm` against factors (full win; more careful algebra, more test surface).

This CIP targets (1) first, with (2) as an optional follow-on once dual-path LL compare is green.

**Out of critical path for this CIP:** FITC (CLI currently errors; code path half-migrated), PITC (unimplemented), non-spherical multi-output. Document hooks so those approximations can opt into the same flag later.

### Flag design

- Member on `CGp`, e.g. `sparseLinalgMode` with values `Legacy` (default) and `StableFactor` (or a bool `useStableSparseLinalg`, default `false`).
- Setter/getter; optional CLI `--sparse-linalg legacy|stable` on `gp` for experiments.
- **Do not** persist in `.model` streams in v1 (avoids I/O version churn). Mode is a runtime / test concern.
- Tenet alignment: keep hot-path BLAS; no new matrix library; asserts on shapes when switching paths.

### Dual-path compare

- Diagnostic or test helper: for a fixed model state, run `updateAD` / log-likelihood / `checkGradients` under both modes and report max abs / relative diffs.
- Extend `diagDtcvar` (or a sibling) and/or add a gated case in `testGp` when fixtures exist.
- Acceptance band: well-conditioned synthetic problems should match within something like `ndlutil::MATCHTOL` or a documented slightly looser band for LL; document any systematic drift.

### What “stable” means here

Not a new approximation — same DTC / DTCVAR mathematics, different factorisation schedule. Same jitter policy (`jitChol`) remains; the stable path should not silently change the mathematical model.

## Implementation Plan

1. **Inventory call sites** that form or consume `invK_uu` in `CGp.cpp` / `CGp.h` for DTC and DTCVAR (likelihood, `updateAD`, `_updateInvK`, gradient blocks). Mark which can use `trsm` immediately vs which still need the inverse for grads.
2. **Add flag + plumbing** (`CGp` member, accessors, optional CLI). Default remains Legacy.
3. **Implement StableFactor forward path** for spherical DTC and DTCVAR: Cholesky of \(K_{uu}\), `trsm` for \(B\), SYRK / existing \(A\) construction without requiring `pdinv(K_uu)` for the LL. Keep Legacy path untouched.
4. **Dual-path harness**: extend `diagDtcvar` (or new `diagSparseLinalg`) to print LL and gradient max-diff between modes on the fixed synthetic problem already used for DTCVAR diagnostics.
5. **Wire a test gate**: script or C++ assert that both modes agree on that problem within tolerance; document how to run it in README / maintainer notes.
6. **Optional follow-on** (separate commits or a later backlog task after Accept): rewrite gradient consumers to avoid `invK_uu`; consider FITC once that approximation is rehabilitated.

## Backward Compatibility

- Default mode = Legacy → existing behaviour, fixtures, and CLI unchanged.
- No stream format change in v1.
- Enabling StableFactor is opt-in for developers and diagnostics until dual-path evidence supports flipping the default (a later decision, not this CIP’s close criteria).

## Testing Strategy

- Re-run existing DTC / DTCVAR diagnostics (`make diagDtcvar`, `scripts/diag_dtcvar_compare.py`) under both modes.
- Dual-path max-diff report on LL and analytic gradients; `checkGradients` must still pass under StableFactor.
- `make run-matio-tests` remains green with default Legacy.
- Ill-conditioned probe (optional): shrink lengthscales / inflate \(M\) / reduce noise and record jitter warnings and LL divergence between modes — informational, not a hard fail until we define a policy.

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md) — strengthens verification for sparse approximations.
- Tenets: assert-and-verify; LAPACK owns linear algebra; flexibility without a performance tax (BLAS `trsm`/`syrk` vs explicit inverse).

## Implementation Status

- [ ] Inventory `invK_uu` consumers for DTC / DTCVAR
- [ ] Add `sparseLinalgMode` flag (default Legacy) + optional CLI
- [ ] StableFactor forward path for spherical DTC / DTCVAR
- [ ] Dual-path diagnostic harness
- [ ] Documented test / tolerance gate
- [ ] (Optional) Gradient path without `pdinv(K_uu)`
- [ ] (Optional) FITC / PITC hooks

## References

- Seeger, Hetzel, Dai, Meissner, Lawrence (2017). [Auto-Differentiating Linear Algebra](https://arxiv.org/abs/1710.08717). Sparse GP section: `potrf` / `trsm` / `syrk`.
- http://inverseprobability.com/publications/auto-differentiating-linear-algebra.html
- `CGp::updateAD`, `CGp::_updateInvK` in `CGp.cpp`
- `backlog/bugs/2026-11-15_dtcvar-numerical-diagnostic.md` — DTCVAR already matches GPy on the toy path; this CIP is hygiene + dual-path verify, not a bugfix
- CIP-0005 — prefer BLAS flags over physical Cholesky flips (complementary)
