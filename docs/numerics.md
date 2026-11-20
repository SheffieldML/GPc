# Numerics and reference fixtures

Decisions that keep the matio reference suite trustworthy against MATLAB-era fixtures.

**Design**: [CIP-0003](../cip/cip0003.md) · [CIP-0005](../cip/cip0005.md) · [CIP-0008](../cip/cip0008.md) · [CIP-0009](../cip/cip0009.md) · [CIP-000A](../cip/cip000A.md) · [CIP-000E](../cip/cip000E.md) · [CIP-000F](../cip/cip000F.md) · [CIP-0012](../cip/cip0012.md) · [CIP-0019](../cip/cip0019.md) · [CIP-001A](../cip/cip001A.md)  
**Requirement**: [REQ-0002](../requirements/req0002_runnable-verification.md) · [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md)

## Adversarial numerical stability ([CIP-0019](../cip/cip0019.md) / [CIP-001A](../cip/cip001A.md))

Separate from matio fixtures: `make run-numerical-stability` builds and runs
`numerical_stability/testNumericalStability`. Contracts are finiteness,
accuracy, and invariance under extreme inputs (GPy CIP-0011 sibling). CI:
`.github/workflows/numerical-stability.yml` publishes a Codecov-style report
(job summary, artifact, sticky PR comment via `report.py`) plus CIP-001A
`metrics.json` trend. Cross-branch outcome diff:
`python3 numerical_stability/compare.py`. Strict xfails are listed in CIP-0019.

**Two layers.** CIP-0019 owns pass/xfail `OUTCOME` cliffs (NaN / invariance /
unexpected fail). CIP-001A adds a parallel `METRIC` stream: value, category,
budget, and pinned cell label. Soft-warn by default on PR trend; do not merge
metrics into `run-matio-tests`. Do not copy OUTCOME cliff tolerances into
`emitMetric` budgets — cliffs detect gross failure; budgets track the category
error mechanism. Tighten budgets only with analysis; never hide a bug by
loosening them.

| Category | Budget (starting) | Typical probes |
|---|---|---|
| `A_machine` | rtol 1e-14 / atol 1e-15 | `logDet` on scaled I; well-conditioned Cholesky reconstruct |
| `B_linalg` | rtol 1e-12 / atol 1e-13 | `pdinv` residual vs cond; `jitChol` residual |
| `C_special` | rtol 1e-10 / atol 1e-12 | `lnCumGaussian` vs `log(Φ)`; mills asymptote |
| `D_analytic` | rtol 1e-8 / atol 1e-9 | kernel symmetry / diag; exact-vs-FITC when close |
| `D_fd` | rtol 1e-5 / atol 1e-5 | FD vs analytic hyperparameter grads |
| `E_invariance` | atol 1e-8 … 1e-6 | translation ΔK; input-rescaling LL |
| `F_model` | rtol 1e-4 / atol 1e-5 | inducing≈X LL gap; predictive variance floor |

Sparse algebra mode for the suite: `GPC_SPARSE_LINALG=stable|legacy` (cell label
becomes `…+stable` / `…+legacy`). Dual-path metric compare:
`make compare-sparse-modes` or `python3 numerical_stability/compare_sparse_modes.py`.

Improvement experiments (in-suite, both modes every run): crowded inducing +
CIP-0013-style protocol stress (`*_stress_stable_finite`,
`*_stress_stable_fd_not_worse`, optional `*_legacy_breaks_stable_ok`), plus a
reduced γ×M dual-path slice. Full γ×M / mpmath remains `make diagSparseStable`.

Optimiser budgets (`-#` / L-BFGS vs SCG vs GPy): see [optimisers.md](./optimisers.md).

## CMatrix ownership ([CIP-0008](../cip/cip0008.md))

`CMatrix` owns its column-major `double* vals` buffer under the Rule of Five: deep copy construction/assignment, move construction/assignment that transfer the buffer, and a destructor that frees it. Do not rely on compiler-generated assignment (that aliases pointers). Prefer `deepCopy` when an existing object must take on another matrix’s values without using `operator=`; moved-from matrices are empty (`vals == NULL`) and may only be destroyed or assigned into.

## Compound / tensor kernel ownership ([CIP-000A](../cip/cip000A.md))

`CComponentKern` stores children in `vector<unique_ptr<CKern>>`. `addKern(const CKern*)` clones into an owned component (caller keeps the argument). `addKern(unique_ptr<CKern>)` takes ownership without cloning — use that for stream and MATLAB factories that allocate. The shallow `CComponentKern` copy constructor is deleted; `CCmpndKern` / `CTensorKern` deep-copy via `addKern` (including the tensor slash / leave-one-out constructor). Model-level owners such as `CGp::pkern` remain raw pointers for now.

## GP / GPLVM working-state caches ([CIP-0009](../cip/cip0009.md))

`CGp` and `CGplvm` still store factorisations and gradient scratch in `mutable` members so `const` likelihood / gradient entry points (required by `COptimisable`) can refresh them. The explicit protocol is:

- Mutators call `invalidateCache()` (or a specific `set*UpToDate(false)`).
- Evaluators call `ensureMUpToDate` / `ensureKUpToDate` / `ensureADUpToDate` / `ensureAlphaUpToDate` before reading cached factors.
- One model instance is single-threaded; do not share a live `CGp` / `CGplvm` across threads.

Intra-evaluation parallelism (GEMM distances via multi-threaded BLAS; optional OpenMP over disjoint workspace rows) is [CIP-0025](../cip/cip0025.md) / [REQ-0013](../requirements/req0013_intra-eval-kernel-parallelism.md) — L4 relative to the L2/L3 contract in [CIP-001F](../cip/cip001F.md). See [docs/build-and-ci.md](./build-and-ci.md) for `GPC_OPENMP` and BLAS nesting. Same-instance concurrent APIs remain out of scope.

## Spherical PITC ([CIP-000F](../cip/cip000F.md))

`CGp::PITC` implements the GPmat spherical path: contiguous `blockEnd` partitions (default matching `gpCreate`), block \(D\) / \(A\) updates, log-likelihood, and `gpCovGrads`. Gate with `matfiles/testGppitc.mat` via `testGaussian("pitc")`. Regenerate with `make USE_MATIO=1 genTestGpPitc && python scripts/gen_testGppitc.py`. FD check: `make diagPitc && ./diagPitc`.
## FITC log-likelihood 2π term

FITC shares the common `-(d N/2) log 2π` tail in `CGp::logLikelihood` with DTC/DTCVAR/PITC. An earlier FITC-only `+ N log 2π` before `L *= -0.5` double-counted that constant (offset \(\tfrac{d N}{2}\log 2\pi\); gradients unaffected). Removed 2026-11-16 — see backlog `2026-11-15_fitc-logtwopi-double-count`. MATLAB-era FITC fixtures may still differ by that amount; `testGp` accepts either convention via `logLikMatchesFixture`.

## Sparse inducing families

`CGp` sparse approximations (DTC / DTCVAR / FITC / PITC) take an inducing family via `--inducing` / `setInducingType`:

| Family | CLI | \(K_{uu}\) | When to use |
|---|---|---|---|
| **points** (default) | `points` | \(k(Z,Z)\) | General case; inducing centres in input space |
| **whitenoise** | `whitenoise` | window Gram \(\langle\psi_m,\psi_n\rangle\) (kernel-free) | Stationary kernels (RBF, Matérn, …) in any \(d\); lengthscale-driven \(\kappa(K_{uu})\) blow-up ([CIP-001C](../cip/cip001C.md)). At window width \(s=\ell/\sqrt2\) (RBF) this reduces to point inducing (equivalence gate); fix \(s\) (default: median NN spacing of centres) to keep \(K_{uu}\) kernel-free |
| **greens** | `greens` | same window Gram as WN | **1-D** causal / Markov kernels (Brownian, integrated Brownian, Exp / Matérn-\(\nu=p+\tfrac12\)); erf / smoothed-ramp \(K_{fu}\) ([CIP-0029](../cip/cip0029.md)) |
| **weightspace** | `weightspace` | ridge Gram in whitened \(\boldsymbol\omega\) (kernel-free; \(\mathrm{diag}=1\)) | Dot-product kernels: **MLP** (finite \(\Sigma\), learn \(w,b\)) or **arccos** order 0 (infinite / angular) / order 1 (soft-ReLU); [CIP-0028](../cip/cip0028.md) |

Fixed window width: `--window-width` (default: median nearest-neighbour spacing of centres; for weight-space ridge, init picks \(s\) from offset spacing under a \(\kappa(K_{uu})\le10^4\) budget). Greens rejects \(d\neq1\). Prefer DTCVAR for the variational bound; points remain the right default when \(K_{uu}\) is well conditioned.

**Weight-space compounds.** Exactly one signal component among `mlp` / `mlpard` / `arccos`, plus optional `white` (contributes 0 to \(K_{fu}\)). Put observation noise in \(\beta\) / `Gaussian_noise.variance` ([CIP-0027](../cip/cip0027.md)). Bias, RBF, or a second signal kernel throws a clear Error. Init: orthogonal direction blocks capped by \(D'=d+1\), grow \(M\) via \(n_{\mathrm{off}}\). Prefer MLP when \(\Sigma\) should be learned; arccos-0 for the angular / infinite-\((w,b)\) limit; arccos-1 for Cho–Saul order-1 / soft-ReLU ridges (capture ~0.4 at the same \(\kappa\) budget where erf reaches ~0.9).

Stability cells: `wn_ls_sweep_*` (point vs white-noise RBF); `greens_var_sweep_*` / `greens_ls_sweep_*` (Brownian variance and Matérn lengthscale vs greens); `ws_ridge_wv_sweep_*` (ridge weight-space: \(\kappa\) flat in `weightVariance`, variance captured, DTCVAR finite under StableFactor/LQ).

## Sparse inducing linear algebra ([CIP-000E](../cip/cip000E.md) / [CIP-0012](../cip/cip0012.md) / [CIP-001B](../cip/cip001B.md))

Runtime flag only (not stored in `.model` files). Toggle with `CGp::setSparseLinalgMode` or `gp … --sparse-linalg stable|lq|legacy` (prefer the long form: short `-L` collides with `--Learn-scales`).

[CIP-000E](../cip/cip000E.md) introduced the dual-path flag and the first StableFactor forward path (trsm for \(V\)). [CIP-0012](../cip/cip0012.md) **upgraded StableFactor in place** to whitened \(C=I+B\Lambda^{-1}B^\top\) with reverse-mode adjoint gradients — 000E is superseded-complete; the flag and `diagDtcvar` harness remain. [CIP-001B](../cip/cip001B.md) adds the LQ default for collapsed bounds.

- **LQ** (default, [CIP-001B](../cip/cip001B.md)): whitened forward \(B = U^{-\top}K_{uf}\); form \(C\) by LQ of \([I,\,\sqrt{\beta}B]\) (DTC/DTCVAR) or \([I,\,\sqrt{\beta}V]\) (FITC) instead of Gram+chol. Upper \(R=L^\top\) lands in `LcholC` for prediction. Gradients reverse through Seeger §4.3 `lqAdjoint` (`CLQAdjoint`) then the CIP-0012 whitening adjoints — not `pdinv(C)`. PITC stays on the StableFactor Gram path under `lq` for now.
- **StableFactor** ([CIP-0012](../cip/cip0012.md)): same whitening; \(C = I + B\Lambda^{-1}B^\top\) via `syrk`+`potrf` (DTC/DTCVAR: \(C=I+\beta BB^\top\); FITC diagonal \(\Lambda\); PITC: \(Q_b=B_b^\top B_b\), \(C=I+\beta\sum B_b D_b^{-1}B_b^\top\)). Reverse-mode grads for DTC/DTCVAR/FITC/PITC use Murray blocked Cholesky / `trsm` adjoints (`CCholAdjoint`). Prediction for all four approximations uses \(U,C\) factors.
- **Legacy**: `pdinv` of \(K_{uu}\) / \(A\) and `gemm(invK_uu, K_uf)` where needed. Keep for dual-path archaeology (`gp … --sparse-linalg legacy`).

Gates:

- Dual-path toy: `make testCGp && ./testCGp` (StableFactor + LQ sections) and `make diagDtcvar && ./diagDtcvar`.
- LQ unit: `make testLQAdjoint && ./testLQAdjoint` (includes ill-conditioned \(K_{uf}K_{fu}\) logdet vs SVD — the large-\(N\) C-side cell, distinct from crowdZ). Offline: `python scripts/lq_sparse_prototype.py`.
- γ×M stress dual-path: `make diagSparseStable && ./diagSparseStable` (DTC/DTCVAR/FITC/PITC).
- Stable vs LQ hard cells: `make diagLqVsStable && ./diagLqVsStable` (crowdZ = \(K_{uu}\)-limited; `largeN_clust_kufkfu_*` = \(K_{uf}K_{fu}\)-limited).
- Numerical-stability suite dual-path metrics: `make compare-sparse-modes` (DTC/DTCVAR/FITC/PITC FD + Legacy↔StableFactor LL/grad gaps).
- Numerical-stability stress: crowded-Z / protocol FITC+DTCVAR both-mode FD ordering; γ×M dual-path slice in CI via `make run-numerical-stability`.
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
