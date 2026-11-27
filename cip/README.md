# Code Improvement Proposals (CIPs)

## Overview

Code Improvement Proposals (CIPs) document design decisions, track implementation, and provide context for substantial changes to GPc.

## Process

1. **Create a New CIP**:
   - Copy `cip_template.md`
   - Name it `cipXXXX.md` with the next hexadecimal number
   - Fill in the proposal; link `related_requirements`

2. **Review**:
   - Share for feedback while status is Proposed
   - Accept before creating backlog implementation tasks

3. **Implementation**:
   - Move to In Progress; tick Implementation Status items as work lands

4. **Completion**:
   - Implemented → Closed after verification; consider documentation compression

## Current CIPs

- [CIP-0001](./cip0001.md)–[CIP-000B](./cip000B.md): see individual files (**Closed**; 000A unique_ptr kernels; 0005 transpose; 0006 macOS CI; …)
- [CIP-000C](./cip000C.md): Finish ARD stubs and import StdPeriodic, Cosine, Brownian (**Closed**)
- [CIP-000D](./cip000D.md): Multi-output covariance via Coregionalize (**Closed**).
- [CIP-000E](./cip000E.md): Stable sparse-GP factorisation path with dual-path compare flag (**Closed** — superseded-complete by [CIP-0012](./cip0012.md))
- [CIP-000F](./cip000F.md): Implement spherical PITC and ship a matio reference fixture (**Closed**)
- [CIP-0010](./cip0010.md): Dedicated testOptimisable suite (**Closed**)
- [CIP-0011](./cip0011.md): Drive near-complete line coverage of `CGp.cpp` (**Closed**)
- [CIP-0012](./cip0012.md): Whitened StableFactor path with reverse-mode adjoint gradients (**Closed** — StableFactor default after soak).
- [CIP-0013](./cip0013.md): GPy-referenced benchmark harness and kernel/BLAS improvements (**In Progress** — reopened 2026-11-22; 2026-11-21 close table withdrawn as cache-reusing). Implements [REQ-0007](../requirements/req0007_competitive-default-build.md).
- [CIP-0014](./cip0014.md): GPy-like path façade for parameter get/set and constraints (**Closed**). Implements [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md). See [docs/param-paths.md](../docs/param-paths.md).
- [CIP-0015](./cip0015.md): Native GPy-tutorial parity via C++ demos and markdown (**Closed**). Implements [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md). Tutorials index: [docs/tutorials/README.md](../docs/tutorials/README.md). Notebooks are checklist only.
- [CIP-0017](./cip0017.md): Psi statistics and Bayesian GP-LVM, GPy-parity subset (**Closed**). CLI: `./bgplvm`.
- [CIP-0018](./cip0018.md): Manifold Relevance Determination, GPy-parity subset (**Closed**). Implements [REQ-000C](../requirements/req000C_manifold-relevance-determination.md). `CMrd` + `testMrd` / `demo_mrd`; see [docs/tutorials/mrd.md](../docs/tutorials/mrd.md).
- [CIP-0019](./cip0019.md): Numerical stability suite, separate from matio reference tests (**Closed**, compressed). Mirrors GPy CIP-0011; see `numerical_stability/` and [`docs/numerics.md`](../docs/numerics.md).
- [CIP-001A](./cip001A.md): Categorised numerical stability metrics and baseline trend (**Closed**). qig-style budgets + Codecov-like Δ vs `main` metrics; extends CIP-0019.
- [CIP-001B](./cip001B.md): LQ factorisation for collapsed VFE/DTC (and FITC) sparse bounds (**Closed**); Seeger et al. 2017 BLR trick on the sparse bound.
- [CIP-001C](./cip001C.md): White-noise inducing features for stationary kernels (kernel-free \(K_{uu}\)) (**Closed**; compressed into [docs/numerics.md](../docs/numerics.md)).
- [CIP-0016](./cip0016.md): Shared JSON/YAML model interchange with GPy and GPmat (**Proposed**). Implements [REQ-000A](../requirements/req000A_shared-model-interchange.md). Orthogonal to CIP-0015 (demos) and CIP-001D (live Python façade).
- [CIP-001D](./cip001D.md): Light Python façade over GPc for GPy users (**Closed** — GPy-form `kern`/`models`/compound `+`; Phase B2 supplied by [CIP-001E](./cip001E.md)). Implements [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md). Orthogonal to CIP-0015 (native tutorials) and CIP-0016 (file interchange). Interactive help → [CIP-002F](./cip002F.md).
- [CIP-001E](./cip001E.md): Interceptable optimiser events and structured model display (**Closed** — #109/#110; compressed into [optimisers.md](../docs/optimisers.md) / [param-paths.md](../docs/param-paths.md)). Implements [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md).
- [CIP-001F](./cip001F.md): L2/L3 concurrency contract — docs, RNG hygiene, BLAS nesting (**Proposed**). Implements [REQ-0010](../requirements/req0010_documented-concurrency-contract.md). Extends CIP-0009 L1; orthogonal to CIP-0013 P7; consumed by CIP-001D/001E docs/GIL.
- [CIP-0020](./cip0020.md): CMake when packaging forces it — keep makefile as blessed native path (**Closed** — #74). Extends [REQ-0001](../requirements/req0001_contemporary-build.md) / [REQ-0007](../requirements/req0007_competitive-default-build.md); supports [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) / [CIP-001D](./cip001D.md) wheels. CI cutover deferred to CIP-0021.
- [CIP-0021](./cip0021.md): CMake CI soak then hard-gate cutover (**Proposed**). Implements [REQ-0001](../requirements/req0001_contemporary-build.md) / [REQ-0002](../requirements/req0002_runnable-verification.md) under Actions CMake; revisits CIP-0020 deferred cutover. Local `make` stays documented.
- [CIP-0022](./cip0022.md): CSV and pods as preferred data path; SVM-light as legacy (**Closed**) Phase E.
- [CIP-0023](./cip0023.md): Resuscitate Probabilistic Point Assimilation (PPA) from Nat King MATLAB (**Closed**). Implements [REQ-0002](../requirements/req0002_runnable-verification.md) / [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md). Tutorial: [`docs/tutorials/ppa_phase0.md`](../docs/tutorials/ppa_phase0.md).
- [CIP-0024](./cip0024.md): EQ / ExpQuad — GPy lengthscale-first SE surface on shared RBF engine (**Closed**). Implements [REQ-0012](../requirements/req0012_eq-expquad-parameterisation.md). See [param-paths.md](../docs/param-paths.md).
- [CIP-0025](./cip0025.md): BLAS-owned distances and OpenMP bulk \(K(r)\) — parallelism without a GPc thread pool (**Closed**).
- [CIP-0026](./cip0026.md): Sparse PPA via CGp-backed EM (DTC first, DTCVAR bound later) (**Closed**).
- [CIP-0027](./cip0027.md): Likelihood-owned observation noise (`likelihood.variance`, alias `Gaussian_noise.variance`); drop auto White/Bias for regression/PPA; **IVM keeps trained White** (softener write-up [`cip0027/ivm_softener.md`](./cip0027/ivm_softener.md)); Phase 0 batched likelihood API; Phase C harden + `testCip0027Perf` ([`phase_c_soak.md`](./cip0027/phase_c_soak.md)); site strip completed in CIP-0034 (**Closed**, compressed). Implements [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md); cites [`docs/target-architecture.md`](../docs/target-architecture.md).
- [CIP-0028](./cip0028.md): Weight-space inducing features for MLP and arc-cosine kernels (**Closed**). Extends [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md) / [CIP-001C](./cip001C.md) beyond stationary white-noise; ridge windows in prior-whitened weight space.
- [CIP-0029](./cip0029.md): Green's-function (causal-root) inducing features for 1-D kernels (**Closed**). Extends [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md) / [CIP-001C](./cip001C.md); sibling of [CIP-0028](./cip0028.md).
- [CIP-002A](./cip002A.md): SVGP — explicit whitened \(q(u)\) with natural gradients (CVI form), Adam for hyperparameters, seeded minibatches; Phase 1 prototype `CSvgp`/`testSvgp` matches `CGp` DTCVAR exactly (**Closed**).
- [CIP-002B](./cip002B.md): Windows MSVC CMake compile smoke on GitHub Actions (**Closed**).
- [CIP-002C](./cip002C.md): Binary wheels so `pip install` needs no local toolchain (**In Progress** — worktree `cip002C-binary-wheels`). Implements [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) installability via cibuildwheel on [CIP-0020](./cip0020.md); preserves [REQ-0007](../requirements/req0007_competitive-default-build.md) OpenBLAS in-wheel. Windows wheels after [CIP-002B](./cip002B.md).
- [CIP-002D](./cip002D.md): CPU quick wins that also prepare for GPUs (optimised default build, cheap checks, BLAS in `lqAdjoint` — 10.3× DTCVAR at N=20k/M=200 — and other O(MN) loops, bulk kernel cross-covariances/gradients, batched SVGP, workspaces, timing artifacts); GPU backend options sketched (**Closed**). Revises CIP-0013 bare-`make` policy; correctness CI uses `CHECKS=full`.
- [CIP-002E](./cip002E.md): Tier CI triggers — Compile/Reference with governance `paths-ignore`; numerics/bench path-filtered; Coverage main/dispatch; Wheels packaging-path + dispatch (**Closed**) soft gate kept.
- [CIP-002F](./cip002F.md): Python-compatible help and docstrings for the `gpc` façade (**Closed**). Implements [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md). Follow-on to [CIP-001D](./cip001D.md); does not reopen modelling scope.
- [CIP-0030](./cip0030.md): Uncollapsed Bayesian GP-LVM — sampled \(q(\mathbf X)\) on the SVGP machinery, for any inducing family (**Closed**); Phase 0 routes `CSvgp` through `CInducing` and is independently shippable.
- [CIP-0031](./cip0031.md): Split `posteriorMeanVar` into latent and predictive (GPy-style) (**Implemented** — worktree `GPc-cip0031` / `cip0031-latent-predictive`). Implements prediction-contract pieces of [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) / [REQ-0002](../requirements/req0002_runnable-verification.md) / [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md). Supersedes backlog `2026-11-19_latent-vs-predictive-meanvar`.
- [CIP-0032](./cip0032.md): Per-unit window widths \(s_m\) for weight-space ridges (**Implemented** — branch `cip0032-per-unit-window-width`). Extends [CIP-0028](./cip0028.md) so each ridge can learn its own softness (MLP-equivalent per-unit scale on the VIK side); tied global \(s\) remains default.
- [CIP-0033](./cip0033.md): Likelihood/inference split to unblock **EP and Laplace** on `CGp` (**Closed**, compressed). Implements [REQ-0019](../requirements/req0019_nonconjugate-inference-ep-laplace.md); spawned from [CIP-0027](./cip0027.md).
- [CIP-0034](./cip0034.md): Strip ADF site caches off `CNoise` into `CNoiseSiteWorkspace` (**Closed**, compressed — closes CIP-0027 residual; validates REQ-0019).
- [CIP-0035](./cip0035.md): Deep GPs by doubly stochastic VI on the SVGP layer core (**Implemented**). Implements [REQ-001A](../requirements/req001A_deep-gaussian-processes.md); stacks the CIP-0030 Phase 0b core. Inducing points by default in hidden layers, learned weight-space ridges optional; identity means with variance \(\chi/(L-1)\). CLI `./deepgp`; tutorial [docs/tutorials/deepgp.md](../docs/tutorials/deepgp.md). Prototype: [`cip0035/prototype/`](./cip0035/prototype/). Worktree: `GPc-cip0035` / `cip0035-deep-gp`.
- [CIP-0036](./cip0036.md): Reflect GPy `GPClassification` capabilities in `gpc` — capability matrix; Laplace predict, naming aliases, notebook (**Implemented**; TestPyPI `0.1.0a6` pending). Builds on CIP-0033/0034; sparse EP deferred.
- [CIP-0037](./cip0037.md): Fix `makeRidgeInit` — one-sided offsets and directions before offsets (**Implemented**). CIP-0028 follow-on (was CIP-0035 draft Phase 0c); unblocks CIP-0035 Phase 2. Re-record fixtures; no legacy flag.
- [CIP-0038](./cip0038.md): GPy `kern` inventory on `gpc` — wrap catalogue, `Sinc`/`ExpQuadCosine`, `Prod`, ICM/coregionalize (**Implemented** — worktree `GPc-cip0038` / `cip0038-gpy-kern-inventory`). Extends [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) / [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md); paths via [CIP-0014](./cip0014.md).
- [CIP-0039](./cip0039.md): PR-time Coverage (path-triggered) and move `CHECKS=full` `testSvgp` / `testSparsePpa` onto the hard gate — amends [CIP-002E](./cip002E.md) demotion (**Proposed**). Implements [REQ-0002](../requirements/req0002_runnable-verification.md).
- [CIP-003A](./cip003A.md): GPy-aligned likelihood zoo in `CNoise` and `gpc.likelihoods` — Student-t, Poisson, Binomial, Exponential, Gamma, SkewGaussian, SkewExponential + Python façade (**Implemented** — worktree `GPc-cip003A` / `cip003A-likelihood-zoo`). Implements [REQ-001B](../requirements/req001B_gpy-aligned-likelihood-zoo.md).
- [CIP-003B](./cip003B.md): Hamiltonian Monte Carlo over `COptimisable` parameters — hyperparameter posterior sampling reusing objective/gradient, priors and transforms (**Implemented**). Implements [REQ-001C](../requirements/req001C_posterior-samples-of-gp-hyperparameters.md). CLI `bgplvm` sampling mode left in backlog.
- [CIP-003C](./cip003C.md): Fixes and gaps found reproducing the deep GP paper experiments — weight-space `(M, d)` construction defect, `DeepGP` façade gaps (**Proposed**). Implements [REQ-0002](../requirements/req0002_runnable-verification.md).
- [CIP-003D](./cip003D.md): Hermite ridge windows as a fixed weight-space inducing family (`inducing='hermite'`), porting the CIP-0035 prototype with quadrature and gradient gates and explicit ask-versus-delivered errors and warnings (**Accepted**). Implements [REQ-0002](../requirements/req0002_runnable-verification.md).
- [CIP-003F](./cip003F.md): Deep GP training speed — batched kernel input/inducing gradients (RBF, linear, MLP, ARD), GEMM variational data term, KL from the Cholesky factor, bulk weight-space ridges, \(O(B)\) SVGP minibatch steps (**Accepted**). Serves [REQ-001A](../requirements/req001A_deep-gaussian-processes.md), [REQ-0015](../requirements/req0015_stochastic-variational-sparse-gp.md).
- [CIP-0041](./cip0041.md): Per-layer `DeepGP` façade — kernel and inducing family per layer, optional scaled MLP initialisation (default unchanged), `mlpard`/`rbfard`, prior-draw Jacobian spread, BLAS thread control (**Accepted**; Phase 1 / B1 implemented). Serves [REQ-001A](../requirements/req001A_deep-gaussian-processes.md); split from CIP-003F.
- [CIP-0040](./cip0040.md): Convolutional covariance functions (`CConvKern`) with inducing patches and weight-space inducing variables over filters; first-layer use in `SparseGPRegression`, `DeepGP` and classification; translation-insensitive and deep convolutional GPs as later phases (**Proposed**). Implements [REQ-001D](../requirements/req001D_convolutional-gps-for-images.md); follows CIP-003F and CIP-0041.

## Creating a Good CIP

A good CIP should:

1. Clearly state the problem being solved
2. Explain the proposed solution in detail
3. Consider alternative approaches
4. Address backward compatibility
5. Include a testing strategy
6. Provide a clear implementation plan
7. Consider potential drawbacks or risks
