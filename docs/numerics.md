# Numerics and reference fixtures

Decisions that keep the matio reference suite trustworthy against MATLAB-era fixtures.

**Design**: [CIP-0003](../cip/cip0003.md) · [CIP-0005](../cip/cip0005.md) · [CIP-0008](../cip/cip0008.md) · [CIP-0009](../cip/cip0009.md)  
**Requirement**: [REQ-0002](../requirements/req0002_runnable-verification.md) · [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md)

## CMatrix ownership ([CIP-0008](../cip/cip0008.md))

`CMatrix` owns its column-major `double* vals` buffer under the Rule of Five: deep copy construction/assignment, move construction/assignment that transfer the buffer, and a destructor that frees it. Do not rely on compiler-generated assignment (that aliases pointers). Prefer `deepCopy` when an existing object must take on another matrix’s values without using `operator=`; moved-from matrices are empty (`vals == NULL`) and may only be destroyed or assigned into.

## GP / GPLVM working-state caches ([CIP-0009](../cip/cip0009.md))

`CGp` and `CGplvm` still store factorisations and gradient scratch in `mutable` members so `const` likelihood / gradient entry points (required by `COptimisable`) can refresh them. The explicit protocol is:

- Mutators call `invalidateCache()` (or a specific `set*UpToDate(false)`).
- Evaluators call `ensureMUpToDate` / `ensureKUpToDate` / `ensureADUpToDate` / `ensureAlphaUpToDate` before reading cached factors.
- One model instance is single-threaded; do not share a live `CGp` / `CGplvm` across threads.

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

## Coverage of library vs drivers

Instrument with `COVERAGE=1`. Strip system paths and `test*.cpp` from lcov uploads so Codecov patch status reflects library changes, not unhit `FAILURE:` branches in test drivers ([CIP-0004](../cip/cip0004.md)).
