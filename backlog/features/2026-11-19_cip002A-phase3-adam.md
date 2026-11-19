---
id: "2026-11-19_cip002A-phase3-adam"
title: "CIP-002A Phase 3: stochastic optimiser interface, CAdam, seeded minibatches"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["002A", "001E"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002A-phase2-gradients"]
tags:
- backlog
- svgp
- adam
- minibatch
---

# Task: CIP-002A Phase 3 — Adam + seeded minibatch loop

## Description

Add a small stochastic interface (`numData`, `objectiveGradBatch`, `getOptParams`/`setOptParams`) and an `CAdam` driver with seeded `mt19937_64` epoch permutation, step schedules, optional interleaved natural-gradient step on \(q\), and outer-iteration progress hooks (CIP-001E observer if present; otherwise a local `std::function` stub that can wire later). Reproducibility test; Gaussian smoke on larger \(N\).

## Acceptance Criteria

- [ ] Same seed ⇒ identical trajectory; different seeds ⇒ same optimum within tolerance on a Gaussian problem
- [ ] Interleaved natgrad + Adam recovers DTCVAR hyperparameter optimum full-batch
- [ ] Progress callback/observer fires each outer iteration without entering objective evaluation

## Related

- CIP: 002A
- REQ: 0015, 000F

## Progress Updates

### 2026-11-19
Task created on CIP acceptance.

### 2026-11-19
Marked Completed after `testSvgp` verification on `cip002A-svgp`.
