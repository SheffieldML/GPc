---
id: "2026-11-19_cip0029-phase-4-stability-docs"
title: "CIP-0029 Phase 4: greens stability cells + numerics docs"
status: "Completed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0029", "0019", "001A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0029-phase-1-brownian", "2026-11-19_cip0029-phase-2-matern"]
tags:
- backlog
- sparse
- inducing
- greens
- numerics
- stability
---

# Task: CIP-0029 Phase 4 — stability cells and docs

## Description

Add numerical-stability suite cells comparing point vs greens inducing (Brownian variance sweep; Matérn lengthscale sweep), and document when to use `points` vs `whitenoise` vs `greens` vs `weightspace` in README / `docs/numerics.md`. Compression of CIP-0029 into formal docs happens after Close (separate checklist).

## Acceptance Criteria

- [x] Stability cells (`greens_var_sweep_*`, `greens_ls_sweep_*`) emit CIP-001A categorised metrics
- [x] Greens path: \(\kappa(K_{uu})\) independent of kernel scale; finite under StableFactor and LQ
- [x] Point-inducing contrast recorded (Matérn ls sweep)
- [x] README and `docs/numerics.md` sparse section mentions greens inducing and 1-D restriction

## Implementation Notes

Phase 3 (integrated Brownian) shipped in the same pass; not a hard dependency of the stability cells.

## Related

- CIP: 0029
- Depends: phase-1-brownian, phase-2-matern

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0029 Accepted; promote to Ready when Phases 1–2 complete.

### 2026-11-19 (later)
Implemented stability cells + docs; status Completed.
