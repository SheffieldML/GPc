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
- [CIP-0013](./cip0013.md): GPy-referenced benchmark harness and kernel/BLAS improvements (**Closed**). Implements [REQ-0007](../requirements/req0007_competitive-default-build.md). Close table: [bench/cip0013_close/](../bench/cip0013_close/).
- [CIP-0014](./cip0014.md): GPy-like path façade for parameter get/set and constraints (**Closed**). Implements [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md). See [docs/param-paths.md](../docs/param-paths.md).
- [CIP-0015](./cip0015.md): Native GPy-tutorial parity via C++ demos and markdown (**Closed**). Implements [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md). Tutorials index: [docs/tutorials/README.md](../docs/tutorials/README.md). Notebooks are checklist only.
- [CIP-0017](./cip0017.md): Psi statistics and Bayesian GP-LVM, GPy-parity subset (**Closed**). CLI: `./bgplvm`.
- [CIP-0018](./cip0018.md): Manifold Relevance Determination, GPy-parity subset (**Closed**). Implements [REQ-000C](../requirements/req000C_manifold-relevance-determination.md). `CMrd` + `testMrd` / `demo_mrd`; see [docs/tutorials/mrd.md](../docs/tutorials/mrd.md).
- [CIP-0019](./cip0019.md): Numerical stability suite, separate from matio reference tests (**Closed**). Mirrors GPy CIP-0011; see `numerical_stability/`.
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
- [CIP-0027](./cip0027.md): Likelihood-owned observation noise (`likelihood.variance`, alias `Gaussian_noise.variance`); drop auto White/Bias for regression/PPA; **IVM keeps trained White** (softener write-up [`cip0027/ivm_softener.md`](./cip0027/ivm_softener.md)); Phase 0 batched likelihood API; Phase C harden + `testCip0027Perf` ([`phase_c_soak.md`](./cip0027/phase_c_soak.md)); EP/sites/`CNoise` data strip deferred (**Closed**, 2026-11-20; #106/#107). Implements [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md); cites [`docs/target-architecture.md`](../docs/target-architecture.md).
- [CIP-0028](./cip0028.md): Weight-space inducing features for MLP and arc-cosine kernels (**Closed**). Extends [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md) / [CIP-001C](./cip001C.md) beyond stationary white-noise; ridge windows in prior-whitened weight space.
- [CIP-0029](./cip0029.md): Green's-function (causal-root) inducing features for 1-D kernels (**Closed**). Extends [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md) / [CIP-001C](./cip001C.md); sibling of [CIP-0028](./cip0028.md).
- [CIP-002A](./cip002A.md): SVGP — explicit whitened \(q(u)\) with natural gradients (CVI form), Adam for hyperparameters, seeded minibatches; Phase 1 prototype `CSvgp`/`testSvgp` matches `CGp` DTCVAR exactly (**Closed**).
- [CIP-002B](./cip002B.md): Windows MSVC CMake compile smoke on GitHub Actions (**Closed**).
- [CIP-002C](./cip002C.md): Binary wheels so `pip install` needs no local toolchain (**In Progress** — worktree `cip002C-binary-wheels`). Implements [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) installability via cibuildwheel on [CIP-0020](./cip0020.md); preserves [REQ-0007](../requirements/req0007_competitive-default-build.md) OpenBLAS in-wheel. Windows wheels after [CIP-002B](./cip002B.md).
- [CIP-002D](./cip002D.md): CPU quick wins that also prepare for GPUs (optimised default build, cheap checks, BLAS in `lqAdjoint` — 10.3× DTCVAR at N=20k/M=200 — and other O(MN) loops, bulk kernel cross-covariances/gradients, batched SVGP, workspaces, timing artifacts); GPU backend options sketched (**Closed**). Revises CIP-0013 bare-`make` policy; correctness CI uses `CHECKS=full`.
- [CIP-002E](./cip002E.md): Tier CI triggers — Compile/Reference with governance `paths-ignore`; numerics/bench path-filtered; Coverage main/dispatch; Wheels packaging-path + dispatch (**Closed**) soft gate kept.
- [CIP-002F](./cip002F.md): Python-compatible help and docstrings for the `gpc` façade (**Closed**). Implements [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md). Follow-on to [CIP-001D](./cip001D.md); does not reopen modelling scope.
- [CIP-0030](./cip0030.md): Uncollapsed Bayesian GP-LVM — sampled \(q(\mathbf X)\) on the SVGP machinery, for any inducing family (**Closed**); Phase 0 routes `CSvgp` through `CInducing` and is independently shippable.
- [CIP-0031](./cip0031.md): Split `posteriorMeanVar` into latent and predictive (GPy-style) (**In Progress** — worktree `GPc-cip0031` / `cip0031-latent-predictive`). Implements prediction-contract pieces of [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) / [REQ-0002](../requirements/req0002_runnable-verification.md) / [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md). Supersedes backlog `2026-11-19_latent-vs-predictive-meanvar`.

## Creating a Good CIP

A good CIP should:

1. Clearly state the problem being solved
2. Explain the proposed solution in detail
3. Consider alternative approaches
4. Address backward compatibility
5. Include a testing strategy
6. Provide a clear implementation plan
7. Consider potential drawbacks or risks
