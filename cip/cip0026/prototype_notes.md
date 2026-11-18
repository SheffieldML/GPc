# CIP-0026 prototype notes (`prototype-sparse-ppa`)

Date: 2026-11-18. Branch experiment before accepting CIP-0026.
(CIP-0025 is the separate BLAS/OpenMP CIP on main.)

## What was built

- [`CSparsePpa`](../../CSparsePpa.h) / [`CSparsePpa.cpp`](../../CSparsePpa.cpp): thin EM driver owning a `CGp` + inner `CGaussianNoise`, outer likelihood via non-owning `CNoise` (probit in tests).
- [`testSparsePpa`](../../testSparsePpa.cpp) + makefile target linking `CGp.o` / `CInducing.o` / chol+LQ adjoints.
- Dense [`CPpa`](../../CPpa.h) / `./ppa` unchanged.

## Gate results (`./testSparsePpa`)

| Gate | Result |
|------|--------|
| DTC \(M=N\), RBF+Bias, vs dense CPpa (init) | PASS (~0) |
| DTC \(M=N\), same, 3 E-steps | PASS (~0) |
| Requested `CGp::FTC` remaps to DTC(\(M=N\)) | PASS |
| DTC \(M=10\) runs, finite `fBar` | PASS |
| Compound kernel **with** white: init match then 2nd E-step drifts | PASS (documents failure mode; drift ≈ 0.053) |
| Raw `CGp::FTC` posterior ≠ PPA `fBar` | PASS (documents β gap; diff ≈ 0.97) |

## Findings (feed the CIP)

1. **M-step / E-step mean really are GP regression on \(\langle f\rangle\) with noise \(1/\beta\).** Wiring `setTargetVals` → `optimise` / `posteriorMean` is enough for the iteration when the approximation is honest.

2. **`CGp::FTC` is not PPA’s M-step.** FTC never adds \(\beta^{-1}I\) to the Gram (noise is expected in a white kernel component). PPA always adds site noise \(1/\beta\) on top of `kern.compute`. Prototype maps “FTC” requests to **DTC with \(M=N\), \(X_u=X\)**.

3. **White kernel breaks DTC \(M=N\) recovery.** `CWhiteKern::compute(X, X2)` is identically zero, so \(K_{uf}\) omits white while \(K_{uu}\) includes it. Even with \(X_u=X\), the sparse path ≠ dense Gram+noise. Phase-0 fixtures use white → cannot be the M=N oracle. Use RBF+Bias and keep observation noise in \(\beta\).

4. **Init fixture match with \(f=0\) is vacuous for the GP mean.** Both paths get \(\langle\bar f\rangle=0\) then identical probit sites. The first *real* posterior check is the second E-step.

5. **Dense monitor stays broken under rank-\(M\) priors.** Prototype does not call `approxLogLikelihood()`; soft signal is `CGp::logLikelihood()` only. A DTCVAR-style PPA bound is still the research-sized piece.

6. **Ownership.** Mutable owned `X` / `f` avoided `const_cast`. Inner Gaussian bias forced to 0; `fixBeta(true)`; inducing fixed after `setInducingToX` when \(M=N\).

7. **Pointer / build plumbing.** `testSparsePpa` needs the full sparse GP object list; production `testPpa` / `ppa` still do not link `CGp`.

## Landed on `cip0026-sparse-ppa` (2026-11-19)

- Phase A: `CSparsePpa` + `testSparsePpa` in makefile `tests`
- Phase B: `./ppa learn -A … -a M` (no white); **`ppaVersion=0.3` stream**
- Phase C: `ppaBound()` = CGp LML + outer noise LL (prefer DTCVAR)
- Phase D: FITC/PITC + free-inducing smoke tests

## Still open (optional follow-up)

- Full \(q(u)\) PPA ELBO recovering dense Cov / \(K^{-1}\) terms
- Sparse `display` / `predict` CLI wiring
- CIP-001C white-noise inducing + PPA sites note
