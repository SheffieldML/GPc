---
id: "2026-11-16_cip0013-bench-kernel-blas"
title: "CIP-0013: benchmark harness and kernel/BLAS improvements"
status: "In Progress"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
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
- [ ] Harness reports median obj+grad time with LL agreement gate (GPy/NumPy when available)
- [ ] P1–P5 done; q=2 protocol meets 1.3× sparse / no full-GP regression
- [ ] Docs + CIP Closed

## Related

- CIP: 0013
- REQ: 0007
- Sibling: CIP-0012 (Murray blocked adjoint; does not block this task)

## Progress Updates

### 2026-11-16

CIP-0013 Accepted → In Progress. Starting harness + P1/P2.
