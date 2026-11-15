---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000D"
last_updated: "2026-11-15"
status: "Proposed"
compressed: false
related_requirements: ["0006"]
related_cips: ["000C"]
tags:
- cip
- kernels
- multioutput
- coregionalize
- gpy
title: "Multi-output covariance via Coregionalize"
---

# CIP-000D: Multi-output covariance via Coregionalize

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-0006](../requirements/req0006_multioutput-covariance.md). It is deliberately separate from [CIP-000C](./cip000C.md) (single-output kernel coverage).

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add intrinsic/linear coregionalization as a first-class `CKern` building block (GPy `Coregionalize`), document how to tensor it with a spatial kernel for multi-output GPs, verify against GPy-generated references, and provide one end-to-end training or evaluation path. Do not port GPy’s full multi-output wrapper zoo in this CIP.

## Motivation

Correlated vector-valued outputs need a cross-output covariance structure. That cannot be obtained by importing more stationary single-input kernels (CIP-000C). GPy’s standard answer is ICM/LCM via `Coregionalize` with \(B = WW^\top + \mathrm{diag}(\kappa)\) on an output-index dimension, tensored with a kernel on the shared inputs. Bringing that contract into GPc keeps ecosystem parity for the common multi-output case while preserving Compose Through Interfaces.

## Detailed Description

### Mathematical form (target)

\[
B = W W^\top + \mathrm{diag}(\kappa), \qquad
k_{\mathrm{ICM}}((x,i),(x',j)) = B_{ij}\, k_{\mathrm{space}}(x,x')
\]

In GPy this is typically `k_space.prod(Coregionalize(...))` with an augmented input that carries the output index. GPc should expose the same idea using existing `CTensorKern` (or an explicit documented composition helper) plus a new `CCoregionalizeKern`.

### Design decisions for this CIP

1. **`CCoregionalizeKern`** — Implements the \(B_{ij}\) factor as a `CKern` on the output-index feature(s). Parameters: `W` (`outputDim × rank`) and `kappa` (`outputDim`), with transforms aligned to GPy (kappa positive / log-exp style via existing `CTransform` machinery). Recompute or cache `B` when parameters change.

2. **Output index encoding** — Adopt GPy’s practical convention: one input column holds the integer output index (0 … \(D-1\)). Document required layout for `X` / `y` stacking (block by output or interleaved). Prefer not inventing a parallel multi-output API inside `CGp` until the kernel path is proven; extend `CGp` only as needed so likelihood and prediction see the stacked design.

3. **Composition** — Primary supported pattern: `CTensorKern` of `{spatial kerns…}` × `CCoregionalizeKern`. If tensor-with-index-column proves awkward given current `inputDim` assumptions, document a thin `CIcmKern` façade that owns space + coregionalize pointers—but prefer reusing `CTensorKern` to avoid a second composition graph (Compose Through Interfaces; avoid special-casing RBF-in-GP).

4. **Rank and size** — Constructor takes `outputDim` and `rank` (default 1). `BOUNDCHECK` / `SANITYCHECK` on index range and shapes. No silent truncation of `W`.

5. **Performance** — Hot path should not rebuild full `B` elementwise without need; cache `B` on parameter updates. Element access remains \(O(\mathrm{rank})\) or \(O(1)\) via cached `B(i,j)`. Do not pull Cython or a second matrix stack; use `CMatrix` / BLAS for `W W^\top` when `outputDim` is non-trivial.

6. **End-to-end path** — Minimum: documented example (CLI flags or small driver) that fits a 2-output toy with ICM and writes/loads a `.model` stream including `coregionalize` (exact type string TBD, suggest `coregionalize`). Full IndependentOutputs / Hierarchical / MultioutputDerivative ports are out of scope.

### Explicit non-goals

- GPy `IndependentOutputs`, `Hierarchical`, `MultioutputKern`, `MultioutputDerivativeKern`, `DiffKern`
- SDE multi-output and ODE/LFM families
- Changing default single-output `gp` behaviour
- CIP-000C kernel list (periodic/cosine/Brownian/ARD)

### Alternatives considered

- **Independent GPs per output only** — Insufficient for REQ-0006; no cross-covariance.
- **Full GPy multioutput package port** — Too large; defer wrappers until ICM kernel + one demo work.
- **Hard-code ICM inside `CGp`** — Violates Compose Through Interfaces; rejected.

### Dependency on CIP-000C

None for Coregionalize itself. Spatial side can use existing `CRbfKern` + `CWhiteKern`. CIP-000C is related only in the “GPy kernel import” product sense (`related_cips`), not a hard blocker.

## Implementation Plan

1. **Design lock** — Confirm X/y stacking convention and stream field names (`outputDim`, `rank`, `W`, `kappa`) in this CIP before coding (update this section if review changes them).
2. **`CCoregionalizeKern`** — Header/implementation: compute, diag, GradParams, GradX (as applicable), stream I/O, factory type `coregionalize`.
3. **Composition + GP wiring** — Documented tensor recipe; any minimal `CGp` / noise changes for stacked multi-output data.
4. **Fixtures** — Generator script from GPy `Coregionalize` (+ tensor with RBF); `testKern` type + gradient checks.
5. **Demo** — One runnable example (synthetic two outputs) and README pointer.
6. **CI** — Include new kern test in matio reference workflow.

## Backward Compatibility

- New kernel type and optional demo only
- Existing single-output models and streams unchanged
- Any new CLI flags for output count / ICM rank must default to current single-output behaviour

## Testing Strategy

- Unit/reference: `testKern` for `coregionalize` alone and tensored with RBF against GPy
- Gradient checks on `W` and `kappa`
- Round-trip stream read/write
- Demo smoke: learn on synthetic multi-output data without assertion failures; optional numerical compare to GPy on the same seed

## Related Requirements

- [REQ-0006](../requirements/req0006_multioutput-covariance.md) — primary
- Verification expectations consistent with [REQ-0002](../requirements/req0002_runnable-verification.md)

## Implementation Status

- [ ] Stacking / stream field names confirmed
- [ ] `CCoregionalizeKern` + factory
- [ ] Tensor composition path documented and tested
- [ ] Minimal GP/demo path for two outputs
- [ ] GPy fixtures + `testKern` + CI

## References

- GPy `GPy.kern.src.coregionalize.Coregionalize`
- GPy examples: multioutput / coregionalization regression
- Álvarez et al., kernels for vector-valued functions / ICM–LCM literature (background only)
- Tenets: Compose Through Interfaces; Assert Invariants; Flexibility Without a Performance Tax; Ecosystem Interoperability
