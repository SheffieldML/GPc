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

- [CIP-0001](./cip0001.md)–[CIP-000B](./cip000B.md): see individual files (0005 transpose; 0006 macOS CI; …)
- [CIP-000C](./cip000C.md): Finish ARD stubs and import StdPeriodic, Cosine, Brownian (**Implemented** — close after CI)
- [CIP-000D](./cip000D.md): Multi-output covariance via Coregionalize (**Implemented** — close after CI). GPy `build_XY` / `ICM` / `LCM`. Implements [REQ-0006](../requirements/req0006_multioutput-covariance.md).
- [CIP-000E](./cip000E.md): Stable sparse-GP factorisation path with dual-path compare flag (**Closed** — superseded-complete by [CIP-0012](./cip0012.md))
- [CIP-000F](./cip000F.md): Implement spherical PITC and ship a matio reference fixture (**Implemented** — close after CI)
- [CIP-0010](./cip0010.md): Dedicated testOptimisable suite (**Closed**)
- [CIP-0011](./cip0011.md): Drive near-complete line coverage of `CGp.cpp` (**Implemented** — close after CI)
- [CIP-0012](./cip0012.md): Whitened StableFactor path with reverse-mode adjoint gradients (**Closed** — StableFactor default after soak).
- [CIP-0013](./cip0013.md): GPy-referenced benchmark harness and kernel/BLAS improvements (**Implemented** on main via #60). Implements [REQ-0007](../requirements/req0007_competitive-default-build.md). Closes without waiting for CIP-0012.
- [CIP-0014](./cip0014.md): GPy-like path façade for parameter get/set and constraints (**Closed**). Implements [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md). See [docs/param-paths.md](../docs/param-paths.md).
- [CIP-0015](./cip0015.md): Native GPy-tutorial parity via C++ demos and markdown (**In Progress**). Implements [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md). Stage 1 tutorials/demos done; Phase A helpers next. Notebooks are checklist only.
- [CIP-0017](./cip0017.md): Psi statistics and Bayesian GP-LVM, GPy-parity subset (**Implemented**). Implements [REQ-000B](../requirements/req000B_bayesian-gplvm-psi.md). CLI: `./bgplvm`.
- [CIP-0018](./cip0018.md): Manifold Relevance Determination, GPy-parity subset (**In Progress** — code + tutorial docs; GPy fixture gate open). Implements [REQ-000C](../requirements/req000C_manifold-relevance-determination.md). `CMrd` + `testMrd` / `demo_mrd`; see [docs/tutorials/mrd.md](../docs/tutorials/mrd.md).
- [CIP-0019](./cip0019.md): Numerical stability suite, separate from matio reference tests (**Implemented** — close after CI). Mirrors GPy CIP-0011; see `numerical_stability/`.
- [CIP-001A](./cip001A.md): Categorised numerical stability metrics and baseline trend (**Closed**). qig-style budgets + Codecov-like Δ vs `main` metrics; extends CIP-0019.
- [CIP-001B](./cip001B.md): LQ factorisation for collapsed VFE/DTC (and FITC) sparse bounds (**Accepted / In Progress**). Extends CIP-0012 whitening: LQ of \([I,A]\) (or \(y\)-augmented) instead of Gram+chol of \(I+AA^\top\); Seeger et al. 2017 BLR trick on the sparse bound.
- [CIP-001C](./cip001C.md): White-noise inducing features for stationary kernels (kernel-free \(K_{uu}\)) (**Implemented** — Phases 1–4; Close after PR/CI). Implements [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md).
- [CIP-0016](./cip0016.md): Shared JSON/YAML model interchange with GPy and GPmat (**Proposed**). Implements [REQ-000A](../requirements/req000A_shared-model-interchange.md). Orthogonal to CIP-0015 (demos) and CIP-001D (live Python façade).
- [CIP-001D](./cip001D.md): Light Python façade over GPc for GPy users (**In Progress** — Phase A done; Phase B FTC smoke green). Implements [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md). Consumes [CIP-001E](./cip001E.md) for live progress / repr. Orthogonal to CIP-0015 (native tutorials) and CIP-0016 (file interchange).
- [CIP-001E](./cip001E.md): Interceptable optimiser events and structured model display (**Accepted**). Implements [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md). C++ `OptimiserEvent`/observer + `ModelDescription`; CIP-001D binds thin Python progress and repr.
- [CIP-001F](./cip001F.md): L2/L3 concurrency contract — docs, RNG hygiene, BLAS nesting (**Proposed**). Implements [REQ-0010](../requirements/req0010_documented-concurrency-contract.md). Extends CIP-0009 L1; orthogonal to CIP-0013 P7; consumed by CIP-001D/001E docs/GIL.
- [CIP-0020](./cip0020.md): CMake when packaging forces it — keep makefile as blessed native path (**In Progress** — Phase 0–A done; B/C open). Extends [REQ-0001](../requirements/req0001_contemporary-build.md) / [REQ-0007](../requirements/req0007_competitive-default-build.md); supports [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) / [CIP-001D](./cip001D.md) wheels.

## Creating a Good CIP

A good CIP should:

1. Clearly state the problem being solved
2. Explain the proposed solution in detail
3. Consider alternative approaches
4. Address backward compatibility
5. Include a testing strategy
6. Provide a clear implementation plan
7. Consider potential drawbacks or risks
