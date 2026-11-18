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

Sparse monitor `ppaBound()` = CGp approximation LML on \(\langle f\rangle\) + outer noise LL at \(\langle\bar f\rangle\) with site variance \(1/\beta\). Prefer DTCVAR for the GP term. Full dense-equivalent Cov / \(K^{-1}\) terms deferred as optional follow-up.

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
