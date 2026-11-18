---
id: "2026-11-19_cip0026-phase-c-dtcvar-bound"
title: "CIP-0026 Phase C: DTCVAR-style sparse PPA monitor bound"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0026"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0026-phase-a-csparseppa"]
tags:
- backlog
- ppa
- dtcvar
- elbo
---

# Task: CIP-0026 Phase C — sparse PPA bound

> Backlog for [CIP-0026](../../cip/cip0026.md) Phase C.

## Description

Sparse monitor started as CGp LML + outer noise LL; superseded by full \(q(u)\) ELBO in `2026-11-19_cip0026-full-qu-elbo` for DTC/DTCVAR.

## Acceptance Criteria

- [x] Sparse path does not call dense `invK` / `fBarfBar` for monitoring
- [x] `CSparsePpa::ppaBound()` documented and tested
- [x] Finite for M&lt;N (DTC and DTCVAR)
- [x] No fake “monotone every iteration” assert

## Related

- CIP: 0026

## Progress Updates

### 2026-11-19

`ppaBound()` landed; soft delta logging only.
