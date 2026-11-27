---
id: "2026-11-27_cip003f-phase3-gemm-data-term-kl"
title: "Phase 3: GEMM data term and KL from Cholesky (A4, A5)"
status: "Ready"
priority: "Medium"
created: "2026-11-27"
last_updated: "2026-11-27"
category: "features"
related_cips: ["003F"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- deepgp
---

# Task: Phase 3: GEMM data term and KL from Cholesky (A4, A5)

> Implements CIP-003F (HOW). See the CIP for the design and gates.

## Description

BLAS-3 data-term q-gradients and site adjoint; KL gradient from the stored Cholesky factor.

## Acceptance Criteria

- [ ] Gates listed for this phase in CIP-003F pass

## Implementation Notes

Work on branch `cip003F-deepgp-training-speed` (worktree `../GPc-cip003F`).

## Related

- CIP: 003F

## Progress Updates

### 2026-11-27
Task created with Ready status when CIP-003F was accepted.
