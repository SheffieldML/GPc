---
id: "2026-11-22_cip003A-phase2-student-poisson"
title: "CIP-003A Phase 2: Student-t, Poisson, Binomial CNoise"
status: "Ready"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["003A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-22_cip003A-phase1-skew-noise"]
tags:
- backlog
- likelihood
- student-t
- poisson
- binomial
---

# Task: CIP-003A Phase 2 — Student-t / Poisson / Binomial

## Description

Port Student-t, Poisson, and Binomial observation models as `CNoise` subclasses with Laplace (and EP quadrature where cheap), stream registration, and GPy/SciPy fixtures.

## Acceptance Criteria

- [ ] `student_t`, `poisson`, `binomial` stream types
- [ ] Gradient checks and density fixtures green
- [ ] Laplace path on `CGp` for at least Student-t and Poisson

## Related

- CIP: 003A
