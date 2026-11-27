# CIP-003E call-site audit (non-test source, excluding CNoise.cpp/.h)

Generated 2026-11-26 from `rg` over the worktree at Phase 1. Counts are lines matching; they locate consumers, not prove per-element assumptions. Re-run the grep when a consumer changes.

| Pattern | Files (matching lines) |
|---|---|
| expectedLogLikelihood | `CSvgp.cpp` (1) |
| laplacePointDerivs | `CLaplaceInference.h` (1), `CLaplaceInference.cpp` (1) |
| getGradInputs | `CDist.h` (1) |
| getNuG | `CIvm.cpp` (1), `CPpa.cpp` (1), `CSparsePpa.cpp` (1) |
| updateSites | `CGplvm.cpp` (2), `CIvm.cpp` (1), `CGp.cpp` (1) |
| site caches (get/set Mu/VarSigma) | `CBayesGplvm.h` (2), `CMrd.h` (2), `CIvm.cpp` (14), `CSviBayesGplvm.h` (2), `CPpa.cpp` (6), `bgplvm.cpp` (3), `CSviBayesGplvm.cpp` (1), `CSparsePpa.cpp` (4), `python/src/wrap_gplvm.cpp` (2) |
| logLikelihood(...) | `CBayesGplvm.h` (1), `CBayesGplvm.cpp` (3), `CGplvm.cpp` (2), `CGplvm.h` (2), `CGp.cpp` (3), `CIvm.cpp` (4), `CGp.h` (1), `CIvm.h` (2), `CInterchange.cpp` (3), `CPpa.h` (2), `CInterchange.h` (1), `CMrd.cpp` (3), `CMrd.h` (1), `CMltools.cpp` (4), `CSparsePpa.cpp` (2), `CPpa.cpp` (6), `CModelDescription.h` (1), `ppa.cpp` (3), `gp.cpp` (3), `ivm.h` (1), `CMltools.h` (4), `COptimisable.h` (2), `ppa.h` (1), `python/src/wrap_interchange.cpp` (1), `python/src/wrap_gplvm.cpp` (5), `ivm.cpp` (4) |
| likelihoods(...) | `CIvm.h` (1), `CIvm.cpp` (2), `CPpa.h` (1), `CPpa.cpp` (2), `ppa.cpp` (2), `gp.cpp` (1), `ivm.cpp` (2), `python/src/wrap_gp.cpp` (1) |
| getOutputDim (latent=output assumption) | `CBayesGplvm.h` (1), `CDeepGp.h` (1), `CDeepGp.cpp` (5), `CGplvm.h` (1), `CGplvm.cpp` (11), `CDataModel.h` (1), `CPpa.h` (1), `CGp.cpp` (111), `CIvm.cpp` (39), `CPpa.cpp` (6), `CGp.h` (9), `CSviBayesGplvm.h` (1), `CKern.h` (2), `CInterchange.cpp` (5), `CMrd.cpp` (2), `CMltools.cpp` (33), `CSparsePpa.cpp` (1), `ppa.cpp` (7), `CSvgp.h` (1), `python/src/wrap_gplvm.cpp` (3), `bgplvm.cpp` (1), `gp.cpp` (5), `ivm.cpp` (7), `python/src/wrap_deepgp.cpp` (2), `gplvm.cpp` (1) |

## Findings that change the CIP

- `updateSites` / `getNuG` are also called from `CGplvm` (and through it the Bayesian GPLVM and MRD models), not only IVM/PPA/`CGp`. Those models must refuse block likelihoods (`requireScalarBlocks`) unless a later CIP generalises them.
- The ADF/IVM path is stateful: it reads `CNoise` site caches (`getMu`, `getVarSigma`) and the stored target. Phase 1 adds the explicit-argument `cavityMomentsElement` form behind `getGradInputs` for gaussian, probit, ncnm and ordered; the zoo likelihoods' `getGradInputs` is a Laplace-point approximation, so they do not advertise `CapCavityMoments`.
- Output count is used as the latent count in `CGp.cpp` (about 110 uses), `CIvm.cpp`, `CMltools.cpp`; one-hot N x K targets leave all of these valid.
- The makefile's header dependencies are incomplete: adding virtual functions to `CNoise` left stale objects that segfaulted `testCGp` until a clean rebuild. Phase 1 changes should be validated after `rm *.o`.
