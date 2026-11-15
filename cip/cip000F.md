---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000F"
last_updated: "2026-11-15"
status: "Proposed"
compressed: false
related_requirements: ["0002"]
related_cips: ["000E"]
tags:
- cip
- sparse
- pitc
- gp
- verification
- testing
title: "Implement spherical PITC and ship a matio reference fixture"
---

# CIP-000F: Implement spherical PITC and ship a matio reference fixture

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Finish the partially independent training conditional (PITC) approximation on the spherical Gaussian path in `CGp`, then add `matfiles/testGppitc.mat` and enable it in the matio `testGp` gate — matching the DTCVAR fixture workflow already on this branch.

**Which requirements does this CIP address?** [REQ-0002](../requirements/req0002_runnable-verification.md) (runnable reference verification for sparse GP approximations).

## Motivation

`CGp` enumerates `PITC` and `gp` accepts `-A pitc`, but `updateAD` and `logLikelihood` throw `NotImplementedError`. After enabling FTC/DTC/DTCVAR/FITC under matio, PITC is the only named approximation with no executable path and no fixture. Shipping a fixture alone would fail immediately; the implementation and the reference artefact belong together.

This also supports **Assert Invariants and Verify Against Reference** and **Compose Through Interfaces**: PITC should work through the existing `CGp` approximation switch, not a fork of the model.

## Detailed Description

### Current state

- `CGp::PITC` is in the approximation enum (`approxInt = 3`).
- Spherical PITC in `updateAD` / `logLikelihood` throws `NotImplementedError` (`CGp.cpp`).
- Gradient and prediction switches mention PITC; several arms are incomplete or share DTC/FITC placeholders with comments that FITC/PITC need extra terms.
- GPmat implements PITC via `model.blockEnd` contiguous blocks over the training set (`gpUpdateAD.m` / `gpLogLikelihood.m`).
- No `matfiles/testGppitc.mat`. `testGp` deliberately skips `pitc`.

### Scope (in)

1. **Spherical Gaussian PITC only** — same spherical restriction as the working DTC/DTCVAR/FITC paths.
2. **Block structure** — expose / wire `blockEnd` (or equivalent) so training points are partitioned into contiguous blocks, matching GPmat’s default blocking from `gpCreate`.
3. **Core numerics** — implement `updateAD`, `logLikelihood`, and `logLikelihoodGradient` for PITC so `checkGradients` passes on a small synthetic problem.
4. **I/O** — stream (and matio where needed) round-trip of approximation type and block metadata so `testGp` stream checks work.
5. **Fixture** — generate `matfiles/testGppitc.mat` from the DTC companion data (or a smaller synthetic set) via Octave/GPmat + C++ stamp, following `scripts/gen_testGpdtcvar.sh`.
6. **Gate** — enable `testGaussian("pitc")` in `testGp` with the same `gpFixtureTol` / \(N/2\log(2\pi)\) LL convention used for the other approximations.

### Scope (out)

- Non-spherical / heteroscedastic PITC.
- Changing the default approximation in `gp`.
- Reworking DTC/FITC factorisation ([CIP-000E](cip000E.md)); PITC may *use* a dual-path flag later but must not block on 000E.
- Full prediction / active-set selection UX polish beyond what `testGp` needs.

### Design notes

- Prefer matching GPmat’s block-loop structure so LL/grads compare cleanly before C++-stamping the fixture.
- Keep LAPACK in the inner block factorisations (`jitChol`, triangular solves); no hand-rolled dense inverse.
- If block metadata is missing today on `CGp`, add the minimum fields and stream keys with a version bump if old `.model` files must remain loadable (default blocks when absent).
- Cross-BLAS fixture compares stay at `gpFixtureTol` (abs 1.0), not `MATCHTOL`.

## Implementation Plan

1. **Inventory** — Map every `PITC` case in `CGp.cpp` / `CGp.h` and list what still throws or falls through.
2. **Blocks** — Define default `blockEnd` construction (GPmat-compatible) and persistence.
3. **Likelihood path** — Implement spherical `updateAD` + `logLikelihood` against GPmat on a tiny problem; record residual.
4. **Gradients** — Implement `logLikelihoodGradient` for PITC; pass `checkGradients` / finite-diff diagnostic.
5. **Fixture** — Octave generator + `genTestGpDtcvar`-style C++ stamp → `matfiles/testGppitc.mat`.
6. **Gate** — Enable PITC in `testGp`; confirm `make USE_MATIO=1 run-matio-tests` on macOS and Linux CI.

## Backward Compatibility

- Existing FTC/DTC/DTCVAR/FITC behaviour unchanged.
- New stream fields for blocks must default safely when reading older `.model` files.
- CLI `-A pitc` becomes usable instead of failing mid-update; document that non-spherical remains unimplemented.

## Testing Strategy

- Unit-scale diagnostic (like `diagDtcvar`) for PITC: finite-diff grads + optional GPmat/GPy compare.
- `matfiles/testGppitc.mat` under matio `testGp` (params `MATCHTOL`; grads/ll `gpFixtureTol`; LL allows \(N/2\log(2\pi)\) convention gap).
- Stream round-trip after `updateM()`.
- Full `run-matio-tests` on Ubuntu and macOS CI.

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md) — sparse GP approximations must be verifiable under the matio gate.

## Implementation Status

- [ ] Inventory PITC stubs and block-state gaps
- [ ] Implement spherical PITC `updateAD` / LL / gradients
- [ ] Persist blocks in stream (version-safe defaults)
- [ ] Add regenerator + `matfiles/testGppitc.mat`
- [ ] Enable `pitc` in `testGp` / matio CI

## References

- Quiñonero-Candela & Rasmussen, “A Unifying View of Sparse Approximate Gaussian Process Regression,” JMLR 2005 (PITC).
- GPmat: `gpUpdateAD.m` / `gpLogLikelihood.m` (`case 'pitc'`), `gpCreate.m` (`blockEnd`).
- In-tree: DTCVAR fixture path (`scripts/gen_testGpdtcvar.sh`, `backlog/bugs/2026-11-15_dtcvar-numerical-diagnostic.md`).
- Related: [CIP-000E](cip000E.md) (stable sparse factorisation dual-path; optional follow-on for PITC).
