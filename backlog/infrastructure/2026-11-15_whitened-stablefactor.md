---
id: "2026-11-15_whitened-stablefactor"
title: "Whitened StableFactor + adjoint grads (CIP-0012)"
status: "Proposed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["0012"]
owner: "Neil Lawrence"
dependencies: ["2026-11-15_fitc-logtwopi-double-count"]
tags:
- backlog
- sparse
- numerics
- infrastructure
---

# Task: Implement CIP-0012

## Description

Accept and implement [CIP-0012](../../cip/cip0012.md): upgrade `SparseLinalgStableFactor` to the whitened \(C = I + B\Lambda^{-1}B^\top\) forward path with reverse-mode Cholesky/`trsm` adjoints for spherical DTC/DTCVAR/FITC/PITC; keep Legacy default until the mpmath soak gate; then flip default.

Prefer fixing the FITC \(2\pi\) double-count bug first or in the same PR train so LL comparisons are not offset-confused.

## Acceptance Criteria

- [ ] CIP-0012 Accepted
- [ ] Whitened forward + adjoint grads; mpmath gate recorded
- [ ] Prediction whitened; docs; CIP-000E Closed
- [ ] Default flip only after soak checkbox
- [ ] CIP-0012 Closed after CI

## Related

- CIP: 0012
- REQ: 0002
- Bugs: `2026-11-15_fitc-logtwopi-double-count`, `2026-11-15_gp-sparse-linalg-cli-collision`
