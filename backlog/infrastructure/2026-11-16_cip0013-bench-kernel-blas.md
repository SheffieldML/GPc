---
id: "2026-11-16_cip0013-bench-kernel-blas"
title: "CIP-0013: benchmark harness and kernel/BLAS improvements"
status: "In Progress"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-22"
category: "infrastructure"
related_cips: ["0013", "0012"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- benchmarking
- kernels
- blas
- gpy
---

# Task: Implement CIP-0013

## Description

Accept and implement [CIP-0013](../../cip/cip0013.md) / [REQ-0007](../../requirements/req0007_competitive-default-build.md): in-repo GPy-referenced bench harness, OpenBLAS + `RELEASE=1` default Linux path, `GPC_CHECKS`, then bulk kernel / distance work (P3–P5).

## Acceptance Criteria

- [x] CIP-0013 Accepted
- [x] Harness reports median obj+grad time with LL agreement gate (GPy/NumPy when available)
- [ ] Harness perturbs params between repeats (includes `setOptParams` / GPy array apply)
- [ ] P1–P5 done; q=2 protocol meets 1.3× sparse / no full-GP regression *(2026-11-21 close withdrawn; cache-busting rebench required)*
- [ ] Docs + CIP Closed

## Related

- CIP: 0013
- REQ: 0007
- Sibling: CIP-0012 (Murray blocked adjoint; does not block this task)

## Progress Updates

### 2026-11-16

CIP-0013 Accepted → In Progress. Starting harness + P1/P2.

### 2026-11-16 (later)

P1/P2 + `benchGPc` on PR #60. P3: direct-loop `dist2Row`/`dist2Col` landed. Next: P4 `CStationaryKern` + RBF reparent.

### 2026-11-16 (P4 slice)

`CStationaryKern` base (`scaleFactor` / `K_of_r`); `CRbfKern` reparented; `CCmpndKern::compute` restored; `CGp::_updateK` uses bulk `compute`. Remaining stationary reparent + P5 workspace still open.

### 2026-11-22

2026-11-21 close used cache-reusing timings (params unchanged between repeats). CIP-0013 reopened. Next: delivery branch for harness fidelity + CoE + REQ-0007 status + reference-Mac rebench; then either meet 1.3× or revise the bar.

### 2026-11-22 (profile — ordered fixes)

Honest sparse gap is structural, not language: (1) harness perturb — this delivery; (2) LQ default / drop `dorglq`+redundant work — CIP-001B; (3) stop Legacy `A`/`Ainv` in whitened `updateAD` — CIP-0012/001B; (4) reuse `Kuf`/`Kuu` in `inducingGradZ` — CIP-002D; (5) bulk vectorised `exp` — CIP-0013. StableFactor already ≈ GPy on VarDTC 10k/200 when measured with param changes.
