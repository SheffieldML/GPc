---
id: "2026-11-18_cip0021-phase2-em-port"
title: "CIP-0021 Phase 2: port E-step and M-step (scalar β)"
status: "In Progress"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["0021"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0021-phase1-compile-skeleton"]
tags:
- backlog
- ppa
- variational
- em
---

# Task: CIP-0021 Phase 2 — E/M port

## Description

Implement `updateExpectationsfBar` / f̄ second moment from MATLAB `ppaExpectfBar` / `ppaExpectfBarfBar`, fill `mStep` via kernel update (`ppaUpdateKernel` path), and wire `init` / `updateK` / `updateInvK` with LAPACK. First ship: scalar β, non-variational kern, `updateBeta=0`.

## Acceptance Criteria

- [x] No `NotImplementedError` on E-step for probit + scalar β
- [ ] `mStep` optimises kernel parameters (or documented no-op only if kernIters=0) — stub calls `runDefaultOptimiser`; needs after-EM fixture gate
- [x] Dense \(K\) / \(K^{-1}\) via Cholesky / `pdinv`
- [x] Live path has no IVM active-set selection API

## Related

- CIP: 0021
- Notes: `cip/cip0021/symbol_map.md`

## Progress Updates

### 2026-11-18
E-step + LL match `testdata/ppa/phase0_*_init` (`./testPpa` 0 failures). Remaining: harden M-step vs `phase0_*_after` fixtures.
