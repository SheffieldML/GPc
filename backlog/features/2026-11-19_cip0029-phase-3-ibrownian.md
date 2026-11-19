---
id: "2026-11-19_cip0029-phase-3-ibrownian"
title: "CIP-0029 Phase 3: integrated Brownian / smoothed-ReLU greens"
status: "Completed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0029"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0029-phase-1-brownian"]
tags:
- backlog
- sparse
- inducing
- greens
- brownian
---

# Task: CIP-0029 Phase 3 — integrated Brownian greens path

## Description

Ship `CIntegratedBrownianKern` (`ibrownian`) with causal-root greens features \(g(u)=\sigma u\mathbf{1}_{u\ge0}\) (Gaussian-smoothed ramp / soft-ReLU units), grads, trapezoid gate, and DTCVAR smoke under `--inducing greens`.

## Acceptance Criteria

- [x] `CIntegratedBrownianKern` covariance \(k(s,t)=\sigma^2(s^2 t/2-s^3/6)\) (same-sign 1-D)
- [x] Closed-form `greensCross*` + FD-checked grads
- [x] `testCGp` Phase 3 section green

## Related

- CIP: 0029

## Progress Updates

### 2026-11-19
Implemented with Phase 4; status Completed.
