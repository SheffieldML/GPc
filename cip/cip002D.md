---
author: "Neil Lawrence"
created: "2026-11-19"
id: "002D"
last_updated: "2026-11-19"
status: "Implemented"
compressed: false
related_requirements: ["0013", "0007", "0010", "0015", "0002", "0016"]
related_cips: ["0013", "0009", "0012", "001B", "002A", "0028", "001F", "0020", "002B", "002C"]
tags:
- cip
- performance
- blas
- build
- checks
- kernels
- svgp
- gpu
title: "CPU quick wins that also prepare for GPUs; GPU backend options sketched"
---

# CIP-002D: CPU quick wins that also prepare for GPUs; GPU backend options sketched

> CIPs describe HOW to achieve requirements (WHAT). Part A lists changes worth doing for CPU performance regardless of any GPU plan; each one is also a prerequisite for a GPU port. Part B sketches the larger GPU issues and options so a later requirement can choose between them. Part B commits to no GPU work.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Profiling `main` (3935b40, 2026-11-19) shows that most CPU time in the sparse paths goes to work that is matrix-shaped but done as scalar loops through `getVal` / `setVal`, and that the default build is unoptimised. Replacing three hand-written loops in `lqAdjoint` with `gemm` calls makes DTCVAR objective+gradient evaluation **10.3× faster at \(N=20\,000, M=200\)**, with identical log-likelihoods and passing adjoint tests. The default `make` / CMake build is **5–14× slower** than `RELEASE=1`.

**Part A** fixes these patterns in eight quick wins:
1. optimise the default build;
2. make checks cheap;
3. BLAS in `lqAdjoint`;
4. BLAS in the other O(MN) loops of `gpCovGrads`;
5. bulk kernel cross-covariances and gradients;
6. batched SVGP and model code;
7. reusable workspaces;
8. timing gates.

**Part B** records what stands between GPc and a GPU:
- matrices held in host memory;
- an element-wise kernel interface;
- hand-written adjoints;
- FP64 throughout;
- small-operation granularity;
- toolchain and CI.

It compares three routes: a native backend, GPc as the CPU reference with GPU training in JAX/PyTorch, and a drop-in GPU BLAS.

## Motivation

### Evidence

All timings are objective+gradient medians from `benchGPc` on a 2-core container with OpenBLAS at `OPENBLAS_NUM_THREADS=1`; \(q=2\), RBF+White. gprof figures cover instrumented GPc code only and exclude time inside BLAS.

**E1 — build defaults** (ms per evaluation):

| case | default (`-ggdb`, checks on) | `-O2` + checks on | `RELEASE=1` (`-O2`, checks off) |
|---|---|---|---|
| FTC, \(N=1000\) | 15.7 | 7.5 | 3.2 |
| DTCVAR, \(N=5000, M=100\) | 2097 | 810 | 152 |
| FITC, \(N=5000, M=100\) | 1765 | 861 | 141 |

Both `make` (without `RELEASE=1`) and CMake (Debug forced when no build type is set) produce the left-hand column.

**E2 — `lqAdjoint` dominates the sparse paths.** In the release DTCVAR profile (\(N=5000, M=100\)), `lqAdjoint` is 92% of instrumented time (109 ms of a 152 ms evaluation). It forms \(L^\top\bar L\), \(\bar Q Q^\top\) and \(\mathrm{copyltu}(M)\,Q\) with triple loops. The \(\bar QQ^\top\) and \(MQ\) products are \(O(m^2 n)\) with \(n=M+N\).

Replacing them with `gemm` (prototype; diff in Appendix A) passes `testLQAdjoint` and `testCGp`, with the same log-likelihood to the printed digits:

| case | loops (ms) | `gemm` (ms) | speed-up |
|---|---|---|---|
| DTCVAR, \(N=5000, M=100\) | 152 | 86 | 1.8× |
| FITC, \(N=5000, M=100\) | 141 | 66 | 2.1× |
| DTCVAR, \(N=20\,000, M=200\) | 6761 | 656 | 10.3× |

**E3 — why checked builds are so slow.** In an `-O2` build with checks on, the DTCVAR profile records **2.18 × 10⁹ calls to the out-of-line `ndlCheckBounds(bool)`**, almost all from element accesses inside scalar loops. Most of the cost of checks is therefore one function call per element access in hot loops, not the checks themselves.

**E4 — next hotspots once `lqAdjoint` uses BLAS** (release, \(N=20\,000, M=200\), 787 ms per evaluation under `-pg`):
- `CGp::gpCovGrads`: 72 ms per evaluation of its own time, mostly \(O(MN)\) accessor loops: block copies into `Abar`, \(\mathrm{tr}(C^{-1}BB^\top)\) and \(\|B\|_F^2\) as element sums, and rank-1 updates of `Lbar`.
- `CRbfKern::getGradParams` (rectangular): 58 ms. It recomputes distances and `exp` element by element.
- `CKern::dist2Active`: \(2.4\times10^7\) virtual calls, each re-testing `isDefaultActiveDims()`.
- `CRbfKern::getGradX`: one call per inducing point.
- `CMatrix::memAllocate`: about \(10^4\) allocations per evaluation (120,775 over 12 evaluations in the \(N=5000, M=100\) profile).

**E5 — SVGP model code** (`CSvgp`, \(N=5000, M=100\), release):
- **Marginals.** The full-data marginals and ELBO take 9.7 ms point by point, against 2.9 ms batched: one `trsm` for \(A=L^{-1}K_{uf}\), one `gemm` for \(\tilde S A\), and column dot products. The values are identical.
- **Cross-covariance.** Building \(K_{uf}\) plus the `trsm` costs 16.8 ms, dominated by the element-wise RBF cross-covariance.

### Why this belongs in one CIP

Every Part A item is a CPU win on its own. Together they move the hot paths onto bulk BLAS-shaped operations with explicit workspaces, which is exactly the form any accelerator backend needs (Part B). Doing them first keeps the GPU decision cheap and reversible.

On Accept, open backlog for **Phase 1 only** (QW1–QW3). Later phases get their own tasks when Phase 1 is verified, so this CIP does not stay open waiting on SVGP batching or a GPU decision record.

## Detailed Description

### Part A — CPU quick wins

**QW1. Optimised default build (revises CIP-0013 bare-`make` policy).**
- [CIP-0013](./cip0013.md) kept bare `make` as `-ggdb` with exhaustive checks, and `RELEASE=1` as the competitive path. **This CIP revises that:** `make` and CMake default to `-O2 -g`, with cheap checks on (QW2, `GPC_CHECKS=1`).
- `RELEASE=1` / `CMAKE_BUILD_TYPE=Release` keep today's meaning: `-O2 -DNDEBUG`, `GPC_CHECKS=0`.
- A new `CHECKS=full` (CMake `-DGPC_CHECKS=2`) restores today's exhaustive checking (per-element bounds and \(O(n^2)\) property scans).
- **Reference / correctness CI uses `CHECKS=full`**, not the new cheap default. Release / timed builds stay on `RELEASE=1`.
- Coordinate with [REQ-0007](../requirements/req0007_competitive-default-build.md); the published q=2 one-thread protocol stays on `RELEASE=1`.
- The wheels in [CIP-002C](./cip002C.md) must be release builds.

**QW2. Cheap checks.**
- Make `ndlCheckBounds` / `ndlCheckDimension` header-inline with an unlikely-branch hint, so a passing check costs a compare rather than a call.
- Split check levels:
  - `GPC_CHECKS=1`: dimensions and API-boundary invariants;
  - `GPC_CHECKS=2`: per-element bounds and \(O(n^2)\) property scans such as `isSymmetric` / `isTriangular`.
- E3 suggests most of the checked-build penalty disappears with QW2+QW3, even before level splitting.

**QW3. `lqAdjoint` via BLAS.** Replace the three triple loops with `gemm` (Appendix A). Also replace the upper-to-lower copies of \(L\) and \(\bar L\) with a single strided copy, or work with the upper factor directly via `trsm` flags.

**QW4. No \(O(MN)\) accessor loops in model code.** In `CGp::gpCovGrads` and the sparse paths:
- block copies → `setMatrix` / `dcopy`;
- traces of products and Frobenius norms → one `ddot` over the contiguous column-major storage;
- rank-1 updates → `dger` / `dsyr`;
- the \(O(MN)\) `Abar` accumulation → `axpy` over the block.

Add a review rule, and optionally a CI grep, flagging any `getVal`/`setVal` loop over two data-sized dimensions in `CGp*.cpp`, `CSvgp.cpp`, `CSparsePpa.cpp` and `CBayesGplvm.cpp`.

**QW5. Bulk kernel paths for cross-covariances and gradients.** Extending [REQ-0013](../requirements/req0013_intra-eval-kernel-parallelism.md) / [CIP-0013](./cip0013.md) P4:
- **Distances:**
  - rectangular \(r^2 = \|x\|^2 + \|z\|^2 - 2XZ^\top\) by `gemm`, with clamping;
  - use GEMM for large blocks even at \(q=2\) (sparse \(K_{uf}\)); keep the CIP-0025 small-\(n\) scalar knee;
  - hoist the `isDefaultActiveDims()` test out of inner loops.
- **RBF/stationary gradients from cached workspaces:**
  - \(\partial/\partial\ell\) as one fused pass over \(K \odot r^2 \odot \bar K\);
  - \(\partial/\partial Z\) for all inducing points as a `gemm`-shaped contraction (e.g. \(\bar K_\odot X - \mathrm{diag}(\bar K_\odot \mathbf 1) Z\)) instead of \(M\) calls to `getGradX` — **coordinate with CIP-0028** (`gradFromKuuKuf` / ridge features); do not reimplement on this branch while 0028 is In Progress.
- **Dot-product kernels** (`CMlpKern`, linear, and CIP-0028 weight-space / directional features): bulk path = one `gemm` plus an element-wise map — **deferred** until 0028 merges (`cip0028-weightspace-inducing` already edits these APIs heavily).
- Agreement with the element-wise oracle to `MATCHTOL` remains the gate.

**QW6. Batched model code.** `CSvgp` marginals, ELBO data term and natural-gradient sites in BLAS-3 form:
- \(A_B = L^{-1}K_{uB}\) via `trsm`;
- \(\mu = A_B^\top\tilde m\) via `gemv`;
- variances from column dot products of \(A_B\) and \(\tilde SA_B\);
- the precision update \(A_B\,\mathrm{diag}(-2c\,g^v)\,A_B^\top\) via `syrk` on scaled columns.

Compute \(K_{uB}\) per minibatch rather than \(K_{uf}\) for all \(N\); this is required for SVI anyway. The same pattern applies to `CSparsePpa`.

**QW7. Workspaces.** Per-model reusable buffers keyed by the [CIP-0009](./cip0009.md) cache protocol for factors, \(A\), \(\bar K\) and distance workspaces. No `CMatrix` allocation inside per-point or per-inducing-point loops. Track `CMatrix::totalAlloc` per evaluation in `benchGPc` and gate on it.

**QW8. Timing as a gate (reconciles CIP-0013 runner noise).**
- [CIP-0013](./cip0013.md) declined hard timing gates on shared runners as too noisy. QW8 does **not** reopen a blind absolute-threshold gate.
- Extend `benchGPc` with `--kind svgp`, `--kern mlp`, and a checked-build mode. (`gpc_checks=` is already printed; `CHECKS=full` vs default/`RELEASE=1`.)
- Record per-PR timings for a fixed set of cases (E1, E2, E5) in CI on release builds as **artifacts first** (`bench/timing_cases.json`, `bench-timing.yml`).
- Only after a same-runner **self-baseline** (compare to the previous successful run on that workflow, not a laptop number) may CI fail on a large regression, e.g. more than 25% on two consecutive runs. Until that baseline exists, treat regressions as warnings (`compare_timing.py`; `--fail-on-regression` opt-in later).

**Expected combined effect (Phase-1 hypothesis, not established fact):** QW1–QW3 are expected to bring the new default build from E1's left column toward or below today's `RELEASE=1` loop numbers for sparse models once cheap checks and `gemm` land together. **Phase 1 gates on re-measurement of E1/E2**, not on that combined claim. QW4–QW6 target the remaining hotspots in E4–E5.

### Part B — GPU readiness: the larger issues (sketch)

**B1. Memory residency.** `CMatrix` owns a host `double*` and every operation calls BLAS eagerly. Pointing `dgemm_` at a GPU library would copy host↔device on every call, which is slower than the CPU at typical sizes (\(M\lesssim1000\)). A backend needs device-resident storage with explicit synchronisation: either a memory-space tag in `CMatrix` or a separate device matrix type. The CIP-0009 cache protocol already tracks staleness and is the natural hook.

**B2. Kernel evaluation model.**
- `CKern`'s hot-path contract is element-wise (`computeElement` and relatives; 27 such declarations in `CKern.h`); bulk compute exists only for stationary kernels.
- After QW5, the bulk interfaces should be the only hot-path API, with element-wise methods kept as test oracles.
- Device implementations then follow per kernel family:
  - stationary: GEMM + map;
  - dot-product and directional features: GEMM + map;
  - compound: sums.

**B3. Hand-derived adjoints.** `CCholAdjoint`, `CLQAdjoint`, the kernel gradients and the SVGP gradients would each need device versions plus parity tests. An autodiff framework removes this cost (option B below).

**B4. Precision.**
- FP64 throughout; consumer GPUs run FP64 at 1/32–1/64 of FP32 throughput.
- A mixed-precision plan would keep the Cholesky, log-determinants and triangular solves in FP64 and run features and GEMMs in FP32. Whether that is safe depends on conditioning.
- Kernel-free, well-conditioned \(K_{uu}\) (white-noise inducing in [CIP-001C](./cip001C.md); directional VIKs in the [CIP-0028](./cip0028.md) update, \(\kappa(K_{uu})\le10^4\)) is the friendliest case.

**B5. Granularity.** GP workloads at \(M\le1000\) are many small operations, so kernel-launch overhead matters. Throughput needs batching across output dimensions, Monte Carlo samples and deep-GP layers, plus fusion of feature maps with reductions. QW6's batched form is the starting point.

**B6. Toolchain and CI.**
- Today: makefile plus CMake ([CIP-0020](./cip0020.md) / [CIP-002B](./cip002B.md)).
- A GPU backend adds CUDA and/or HIP toolchains.
- Default GitHub Actions runners have no GPUs, so the GPU gates would need self-hosted runners.
- Wheels ([CIP-002C](./cip002C.md)) would multiply by GPU variants.

**B7. Concurrency.** [REQ-0010](../requirements/req0010_documented-concurrency-contract.md) / [CIP-001F](./cip001F.md) define a CPU-thread contract. A GPU backend adds streams and library handles per model instance.

**Options.**

| option | what | cost | fits GPc's strengths |
|---|---|---|---|
| A. Native backend | device matrices, cuBLAS/cuSOLVER (or hipBLAS), device kernels and adjoints, CPU/GPU parity tests | large; multi-CIP | keeps one engine and the tenets; doubles the adjoint maintenance |
| B. CPU reference + GPU in JAX/PyTorch | port the closed forms (e.g. directional VIK features, about 20 lines), autodiff for gradients, `vmap` / batching for samples and layers; GPc fixtures and exact checks become the oracle | moderate | builds on GPc's verified numerics; Python stays outside GPc objectives ([REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)) |
| C. Drop-in GPU BLAS (e.g. NVBLAS intercepting `dgemm_`) | link-time only | minimal | offloads only large GEMMs, paying a transfer each time; little benefit at these sizes |

**Recommendation.** Complete Part A first. When a concrete GPU use case is ready, for example deep directional-VIK GPs (CIP-0028 update §4), raise a requirement and prototype option B before committing to A. B gives a GPU reference quickly and costs nothing if A follows later.

### Tenets

- **Flexibility without a performance tax** — Part A removes scalar-loop and debug-default taxes from hot paths.
- **LAPACK owns the linear algebra** — every rewrite is a BLAS/LAPACK call, not a hand loop.
- **Assert and verify** — element-wise oracles stay as tests, checks become cheap enough to leave on, and timings become gates.
- **Compose through interfaces** — bulk kernel interfaces become the hot-path contract that a device backend would implement.

## Implementation Plan

1. **Phase 1 (small):** QW1 default build, QW2 inline bounds checks and levels, QW3 `lqAdjoint` via BLAS. Gate: all suites green in full-checks and release builds; E1/E2 re-measured (combined-effect claim is a hypothesis until then).
2. **Phase 2 (medium):** QW4 `gpCovGrads` / sparse-path loops; QW5 rectangular distance GEMM and fused stationary gradients. **Defer** MLP / directional / inducing batched-\(\partial Z\) bulk until [CIP-0028](./cip0028.md) merges (that branch owns `CMlpKern` / `ws*` / `CInducing`; it does not touch `gpCovGrads` or `CStationaryKern` GEMM).
3. **Phase 3 (medium):** QW6 batched `CSvgp` (per-minibatch \(K_{uB}\) when \(B\ll N\)); QW7 workspaces and allocation counts. `CSparsePpa` has no SVGP-style per-point marginal loop on main — skipped.
4. **Phase 4 (ongoing):** QW8 benchmark artifacts + self-baseline gate design. Write a GPU decision record (new requirement) when a use case is concrete; Part B's options table is the starting point.

## Backward Compatibility

- The default build becomes optimised with cheap checks. Builds that relied on exhaustive default checks set `CHECKS=full`. `RELEASE=1` is unchanged. This amends CIP-0013's bare-`make` policy; correctness CI moves to `CHECKS=full`.
- BLAS rewrites change rounding (reassociation); results agree with the element-wise paths to `MATCHTOL`, and fixtures are unaffected at their tolerances.
- No stream, CLI or API changes. `benchGPc` gains options only.

## Testing Strategy

- **Equivalence:** each rewritten routine is compared with the retained element-wise implementation (`testLQAdjoint`, `testCholAdjoint`, `testCGp`, `testKern`, `testSvgp`), including near-duplicate inputs and \(M=N\) cases.
- **Check levels:** seeded faults (out-of-bounds index, dimension mismatch, non-symmetric input to a symmetric routine) are caught at the documented level and not below it.
- **Performance:** QW8 timings and allocation counts per case; Phase 1 must reproduce E2's speed-up within noise.
- **Coverage:** existing Codecov gates unchanged; new bulk paths are covered by both the equivalence tests and the benchmark smoke.

## Related Requirements

- [REQ-0013](../requirements/req0013_intra-eval-kernel-parallelism.md) — intra-evaluation BLAS / OpenMP (QW5).
- [REQ-0007](../requirements/req0007_competitive-default-build.md) — competitive default build (QW1, QW8).
- [REQ-0010](../requirements/req0010_documented-concurrency-contract.md) — concurrency contract (B7).
- [REQ-0015](../requirements/req0015_stochastic-variational-sparse-gp.md) — SVGP (QW6).
- [REQ-0002](../requirements/req0002_runnable-verification.md) — verification of rewritten paths.
- [REQ-0016](../requirements/req0016_gpu-training-decision.md) — GPU training route decision (Proposed; Part B).

## Implementation Status

- [x] Evidence gathered (E1–E5) and `lqAdjoint` `gemm` prototype measured
- [x] CIP accepted
- [x] Phase 1: QW1–QW3 (code + docs; `testLQAdjoint` / `testCGp` green under `CHECKS=full` and `RELEASE=1`; E1/E2 re-measure still open)
- [x] Phase 2: QW4–QW5 stationary (MLP/0028 bulk deferred; `testCGp` / `testKern` / `testLQAdjoint` green)
- [x] Phase 3: QW6–QW7 (`CSvgp` batched marginals/sites; workspaces; `CMatrix::totalAlloc` + `benchGPc --kind svgp`; `CSparsePpa` skipped — no analogous per-point marginal hot path)
- [x] Phase 4: QW8 (`--kern mlp`; `bench/run_timing_cases.py` + `compare_timing.py`; warn-only `.github/workflows/bench-timing.yml`) and [REQ-0016](../requirements/req0016_gpu-training-decision.md) Proposed from Part B

## References

- [CIP-0013](./cip0013.md) (bulk stationary kernels; bare-`make` policy revised by QW1), [CIP-0012](./cip0012.md) / [CIP-001B](./cip001B.md) (whitened / LQ sparse factorisations and adjoints), [CIP-0009](./cip0009.md) (cache protocol), [CIP-002A](./cip002A.md) (SVGP), [CIP-0028](./cip0028.md) (weight-space / directional inducing features).
- Seeger et al., arXiv:1710.08717 (LQ adjoint, §4.3).

## Appendix A — `lqAdjoint` prototype diff (measured in E2)

```cpp
  // M = L^T Lbar - Qbar Q^T   (m×m)
  CMatrix M(m, m);
  M.gemm(Llow, LbarLow, 1.0, 0.0, "t", "n");
  M.gemm(Qbar, Q, -1.0, 1.0, "n", "t");
  // copyltu(M) unchanged (O(m^2))
  // S = Qbar + copyltu(M) Q
  CMatrix S(m, n);
  S.deepCopy(Qbar);
  S.gemm(Msym, Q, 1.0, 1.0, "n", "n");
```

This replaces the three triple loops between `// M = L^T Lbar - Qbar Q^T` and `// Wbar = L^{-T} S` in `CLQAdjoint.cpp`.
