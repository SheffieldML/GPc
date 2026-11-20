---
id: "2026-11-20_cip0027-phase-c-harden"
title: "CIP-0027 Phase C: optional harden + soak (White + likelihood variance)"
status: "Completed"
priority: "Low"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0027"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0027-phase-bp-ivm-softener"]
tags:
- backlog
- noise
- deprecation
- soak
- performance
---

# Task: CIP-0027 Phase C — Harden dual-noise path

## Description

Optional `--allow-white-and-noise` for Gaussian regression when both White and likelihood variance are present. Soak; then drop dual interpretation if evidence supports it. Include **algorithm performance** regression gates (FTC/sparse LL+RMSE, IVM White Jaccard), not only helper/unit checks.

## Acceptance Criteria

- [x] Flag (or equivalent) documented for intentional White + likelihood variance
- [x] Soak notes recorded; dual-path behaviour tightened or deferred with rationale
- [x] Algorithm performance regression tests (`testCip0027Perf`) for FTC, sparse DTC, IVM softener

## Related

- CIP: 0027
- Depends on: Phase B′
- Soak: [`cip/cip0027/phase_c_soak.md`](../../cip/cip0027/phase_c_soak.md)

## Progress Updates

### 2026-11-20

Task created on Accept.

### 2026-11-20 (complete)

- `CGp::enforceObservationNoisePolicy` on `optimise` / `logLikelihoodGradient`; default refuse dual free noise.
- CLI `--allow-white-and-noise`; learn path auto-`fixLikelihoodVariance` when `-k white` without allow.
- `testCip0027Perf`: policy + sinc FTC/DTC LL/RMSE floors + IVM Jaccard gates.
- Soak decision: keep harden; keep allow escape hatch (do not delete dual-path entirely).
