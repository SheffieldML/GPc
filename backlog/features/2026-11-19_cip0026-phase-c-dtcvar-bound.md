---
id: "2026-11-19_cip0026-phase-c-dtcvar-bound"
title: "CIP-0026 Phase C: DTCVAR-style sparse PPA monitor bound"
status: "In Progress"
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

Replace dense `approxLogLikelihood` (uses \(K^{-1}\), rank-\(N\)) on the sparse path with a monitor suitable for inducing approximations. Prefer a DTCVAR / VFE-shaped bound over \(q(u)\); until the full derivation lands, expose `CGp::logLikelihood` as an interim M-step monitor and document the gap.

## Acceptance Criteria

- [x] Sparse path does not call dense `invK` / `fBarfBar` for monitoring
- [x] Documented interim monitor: `CSparsePpa::monitorLogLikelihood()` = CGp LML
- [x] Test that monitor is finite for M&lt;N (DTC and DTCVAR)
- [ ] Principled PPA–DTCVAR ELBO derivation and implementation
- [x] No fake “monotone every iteration” assert until bound is proven

## Implementation Notes

- Research-ish: re-derive PPA ELBO under q(u). Prototype notes already flag this as the hard piece.
- Interim: `CGp` LML on ⟨f⟩ with current approx type is acceptable if labelled.

## Related

- CIP: 0026

## Progress Updates

### 2026-11-19

Interim monitor landed; full ELBO remains open (task stays In Progress).
