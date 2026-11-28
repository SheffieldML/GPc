---
id: "2026-11-28_cip0043-phase3-deepgp"
title: "CIP-0043 Phase 3: ReLU in the deep GP (C++ and Python), capability query, benchmark"
status: "Proposed"
priority: "High"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0043", "003F", "0041"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_cip0043-phase2-weightspace-hermite"]
tags:
- backlog
- deep-gp
- relu
- python
---

# Task: CIP-0043 Phase 3 — Deep GP

## Description

Accept `relu` / `reluard` in `deepgp.cpp` and `wrap_deepgp.cpp` `makeLayerKern`. Replace the `kernNames[ell] != "mlp"` guards for `weightspace` and `hermite` inducing with `supportsRidgeWindows()`. `mlp_init='scaled'` refuses `relu` with a typed error naming the option.

## Acceptance Criteria

- [ ] `testDeepGp`: ReLU stack gradient check (as the MLP stacks), a two-layer toy fit, and a mixed per-layer stack (`relu` then `rbf`).
- [ ] Python `DeepGP(kernel='relu' | 'reluard', inducing=…)` works per layer; existing error text for unsupported combinations is unchanged where tests match it.
- [ ] `bench/benchGPc --kind deepgp --kern relu` runs; timing rows added to `docs/deepgp.md`.
- [ ] `testDeepGp` and the Python deep GP tests pass.

## Related

- CIP: 0043, 003F, 0041

## Progress Updates

### 2026-11-28

Task created, Proposed (blocked on Phase 2).
