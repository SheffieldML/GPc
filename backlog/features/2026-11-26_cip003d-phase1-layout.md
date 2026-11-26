---
id: "2026-11-26_cip003d-phase1-layout"
title: "CIP-003D phase 1: Hermite layout, c_k and layout audit"
status: "Ready"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- deepgp
- weightspace
- hermite
---

# Task: CIP-003D phase 1: Hermite layout, c_k and layout audit

## Description

Port `hermite_layout` and `c_k` from `cip/cip0035/prototype/dgp.py` to `CInducingHermite::makeLayout`, with the capacity rule \(\lfloor\binom{D'+k-1}{k}/2\rfloor\), `kmax=9`, and degree-1 orthonormal frames reusing the `makeRidgeInit` Gram-Schmidt blocks. Build the `HermiteLayoutReport` (degree histogram, fill per degree, \(\kappa(K_{uu})\)).

## Acceptance Criteria

- [ ] `CInducingHermite` constructs for every \((M,d)\) in a sweep \(M=1..300\), \(d\in\{1,2,8\}\) or throws E1 with `d`, `M`, total and per-degree capacity
- [ ] Post-construction audit (E5): `numInducing()==M`, odd degrees, unit `V`, \(K_{uu}\) block Cholesky succeeds
- [ ] No path returns fewer than `M` windows (the prototype's silent truncation is not reproduced)
- [ ] Layout deterministic for a given seed
- [ ] W1 and W2 computed from the report; tests that each fires on a constructed case and not on a control

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.
