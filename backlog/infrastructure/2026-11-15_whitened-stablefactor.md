---
id: "2026-11-15_whitened-stablefactor"
title: "Whitened StableFactor + adjoint grads (CIP-0012)"
status: "Ready"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-16"
category: "infrastructure"
related_cips: ["0012"]
owner: "Neil Lawrence"
dependencies: ["2026-11-15_fitc-logtwopi-double-count"]
tags:
- backlog
- sparse
- numerics
- infrastructure
- murray
---

# Task: Implement CIP-0012

## Description

Implement [CIP-0012](../../cip/cip0012.md) (Accepted 2026-11-16): upgrade `SparseLinalgStableFactor` to the whitened \(C = I + B\Lambda^{-1}B^\top\) forward path with reverse-mode Cholesky/`trsm` adjoints for spherical DTC/DTCVAR/FITC/PITC; keep Legacy default until the mpmath soak gate; then flip default.

**`_cholAdjoint` must be Murray Level-3 blocked reverse** (Accept decisions). Unblocked/symbolic forms are oracles only. Do not ship a weaker column-matvec VJP as the production path.

Prefer fixing the FITC \(2\pi\) double-count bug first or in the same PR train so LL comparisons are not offset-confused. CIP-0013 may land harness/kernel work first; this task does not block 0013.

## Acceptance Criteria

- [x] CIP-0012 Accepted (Murray blocked adjoint locked)
- [ ] Whitened forward + adjoint grads; mpmath gate recorded
- [ ] Production `_cholAdjoint` is blocked Murray; oracle parity tests green
- [ ] Prediction whitened; docs; CIP-000E Closed
- [ ] Default flip only after soak checkbox
- [ ] CIP-0012 Closed after CI

## Related

- CIP: 0012
- REQ: 0002
- Bugs: `2026-11-15_fitc-logtwopi-double-count`, `2026-11-15_gp-sparse-linalg-cli-collision`
- GPy evidence: CIP-000E `cholesky_vjp_blocked` / Accept Option C

## Progress Updates

### 2026-11-15

Task opened with CIP-0012 at Proposed.

### 2026-11-16

CIP-0012 Accepted. Murray Level-3 blocked `_cholAdjoint` recorded as Accept constraint (GPy CIP-000E lesson). Status Ready; implementation may follow CIP-0013 kernel/BLAS work.
