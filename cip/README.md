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
- [CIP-000E](./cip000E.md): Stable sparse-GP factorisation path with dual-path compare flag (**Implemented** — superseded-complete by 0012 when that lands)
- [CIP-000F](./cip000F.md): Implement spherical PITC and ship a matio reference fixture (**Implemented** — close after CI)
- [CIP-0010](./cip0010.md): Dedicated testOptimisable suite (**Closed**)
- [CIP-0011](./cip0011.md): Drive near-complete line coverage of `CGp.cpp` (**Implemented** — close after CI)
- [CIP-0012](./cip0012.md): Whitened StableFactor path with reverse-mode adjoint gradients (**Proposed**)
- [CIP-0013](./cip0013.md): GPy-referenced benchmark harness and kernel/BLAS improvements (**Proposed**). Implements [REQ-0007](../requirements/req0007_competitive-default-build.md). Closes without waiting for CIP-0012.
- [CIP-0014](./cip0014.md): GPy-like path façade for parameter get/set and constraints (**Closed**). Implements [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md). See [docs/param-paths.md](../docs/param-paths.md).
- [CIP-0015](./cip0015.md): Native GPy-tutorial parity via C++ demos and markdown (**In Progress**). Implements [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md). Stage 1 tutorials/demos done; Phase A helpers next. Notebooks are checklist only.

## Creating a Good CIP

A good CIP should:

1. Clearly state the problem being solved
2. Explain the proposed solution in detail
3. Consider alternative approaches
4. Address backward compatibility
5. Include a testing strategy
6. Provide a clear implementation plan
7. Consider potential drawbacks or risks
