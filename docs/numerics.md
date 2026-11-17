# Numerics and reference fixtures

Decisions that keep the matio reference suite trustworthy against MATLAB-era fixtures.

**Design**: [CIP-0003](../cip/cip0003.md) · [CIP-0005](../cip/cip0005.md) · [CIP-0008](../cip/cip0008.md) · [CIP-0009](../cip/cip0009.md) · [CIP-000E](../cip/cip000E.md) · [CIP-000F](../cip/cip000F.md) · [CIP-0012](../cip/cip0012.md) · [CIP-0019](../cip/cip0019.md)  
**Requirement**: [REQ-0002](../requirements/req0002_runnable-verification.md) · [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md)

## Adversarial numerical stability ([CIP-0019](../cip/cip0019.md))

Separate from matio fixtures: `make run-numerical-stability` builds and runs
`numerical_stability/testNumericalStability`. Contracts are finiteness,
accuracy, and invariance under extreme inputs (GPy CIP-0011 sibling). CI:
`.github/workflows/numerical-stability.yml` publishes a Codecov-style report
(job summary, artifact, sticky PR comment via `report.py`). Cross-branch
outcome diff: `python3 numerical_stability/compare.py`. Strict xfails are
listed in CIP-0019.

Optimiser budgets (`-#` / L-BFGS vs SCG vs GPy): see [optimisers.md](./optimisers.md).

## CMatrix ownership ([CIP-0008](../cip/cip0008.md))

`CMatrix` owns its column-major `double* vals` buffer under the Rule of Five: deep copy construction/assignment, move construction/assignment that transfer the buffer, and a destructor that frees it. Do not rely on compiler-generated assignment (that aliases pointers). Prefer `deepCopy` when an existing object must take on another matrix’s values without using `operator=`; moved-from matrices are empty (`vals == NULL`) and may only be destroyed or assigned into.

## GP / GPLVM working-state caches ([CIP-0009](../cip/cip0009.md))

`CGp` and `CGplvm` still store factorisations and gradient scratch in `mutable` members so `const` likelihood / gradient entry points (required by `COptimisable`) can refresh them. The explicit protocol is:

- Mutators call `invalidateCache()` (or a specific `set*UpToDate(false)`).
- Evaluators call `ensureMUpToDate` / `ensureKUpToDate` / `ensureADUpToDate` / `ensureAlphaUpToDate` before reading cached factors.
- One model instance is single-threaded; do not share a live `CGp` / `CGplvm` across threads.

## Spherical PITC ([CIP-000F](../cip/cip000F.md))

`CGp::PITC` implements the GPmat spherical path: contiguous `blockEnd` partitions (default matching `gpCreate`), block \(D\) / \(A\) updates, log-likelihood, and `gpCovGrads`. Gate with `matfiles/testGppitc.mat` via `testGaussian("pitc")`. Regenerate with `make USE_MATIO=1 genTestGpPitc && python scripts/gen_testGppitc.py`. FD check: `make diagPitc && ./diagPitc`.
## FITC log-likelihood 2π term

FITC shares the common `-(d N/2) log 2π` tail in `CGp::logLikelihood` with DTC/DTCVAR/PITC. An earlier FITC-only `+ N log 2π` before `L *= -0.5` double-counted that constant (offset \(\tfrac{d N}{2}\log 2\pi\); gradients unaffected). Removed 2026-11-16 — see backlog `2026-11-15_fitc-logtwopi-double-count`. MATLAB-era FITC fixtures may still differ by that amount; `testGp` accepts either convention via `logLikMatchesFixture`.

## Sparse inducing linear algebra ([CIP-000E](../cip/cip000E.md) / [CIP-0012](../cip/cip0012.md))

Runtime flag only (not stored in `.model` files). Toggle with `CGp::setStableSparseLinalg` or `gp … -L stable`.

- **StableFactor** (default, [CIP-0012](../cip/cip0012.md)): whitened forward \(B = U^{-\top}K_{uf}\), \(C = I + B\Lambda^{-1}B^\top\) (DTC/DTCVAR: \(C=I+\beta BB^\top\); FITC diagonal \(\Lambda\); PITC: \(Q_b=B_b^\top B_b\), \(C=I+\beta\sum B_b D_b^{-1}B_b^\top\)). Reverse-mode grads for DTC/DTCVAR/FITC/PITC use Murray blocked Cholesky / `trsm` adjoints (`CCholAdjoint`). Prediction for all four approximations uses \(U,C\) factors.
- **Legacy**: `pdinv` of \(K_{uu}\) / \(A\) and `gemm(invK_uu, K_uf)` where needed. Keep for dual-path archaeology (`gp … -L legacy`).

Gates:

- Dual-path toy: `make testCGp && ./testCGp` (StableFactor section) and `make diagDtcvar && ./diagDtcvar`.
- γ×M stress dual-path: `make diagSparseStable && ./diagSparseStable` (DTC/DTCVAR/FITC/PITC).
- Maintainer mpmath (pin `mpmath`, dps 60): after `./diagDtcvar`, run `python scripts/sparse_mp_reference.py diag_dtcvar_state.txt --approx dtcvar`.

## Reference mismatches cleared in CIP-0003

Three cases were deferred from the first matio gate and then fixed so Reference tests could hard-fail:

1. **Tensor kernels** — slash/copy constructors shallow-copied then re-`addKern`, so leave-one-out gradient products used the wrong components. `K` matched; gradients did not. Fixed constructors; `testType("tensor")` re-enabled.
2. **Null-category noise** — MATLAB uses `sigma2 = eps`; C++ had hard-coded `1e-6`. Align with `ndlutil::EPS`; re-enable ncnm in `testNoise`.
3. **Matrix / LAPACK** — `syev` comparisons are sign-invariant; ill-conditioned TRSM fixtures were refreshed for modern LAPACK; `sysv` bound-check crash after `symv` was fixed. `testMatrix` is part of the hard gate again.

## Transpose and ACM Fortran ([CIP-0005](../cip/cip0005.md))

- In-place rectangular transpose is **ACM Algorithm 513** (`DTRANSR` / `dtransr_`). Algorithms 467 (`DXPOSE`) and 380 (`DTRANS`) were unused/broken and were removed.
- Dedicated transpose tests (square, tall, wide, vectors, coprime and non-coprime rectangles, flag behaviour) run in the Reference gate. Square `trans()` via Cholesky in `testChol` is not a substitute for that suite.
- Prefer BLAS/LAPACK `trans` / `uplo` flags over physically flipping Cholesky factors in `CGp` / `CGplvm` when an upper factor can stay upper.
- A blocked C++ square-transpose rewrite is measurement-gated; most model sites already hit the `M == N` fast path inside `DTRANSR`.

## Parameter paths ([CIP-0014](../cip/cip0014.md))

GPy-style `getByPath` / `setByPath` / `fix` on kernels and `CGp`, plus `gp learn -P/--set`. See [param-paths.md](param-paths.md).

## Coverage of library vs drivers

Instrument with `COVERAGE=1`. Strip system paths and `test*.cpp` from lcov uploads so Codecov patch status reflects library changes, not unhit `FAILURE:` branches in test drivers ([CIP-0004](../cip/cip0004.md)).
