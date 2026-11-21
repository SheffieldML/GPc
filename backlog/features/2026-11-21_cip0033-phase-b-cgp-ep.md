---
id: "2026-11-21_cip0033-phase-b-cgp-ep"
title: "CIP-0033 Phase B: wire EP into CGp FTC + GPy fixture"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0033"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0033-phase0-api-inventory"]
tags:
- backlog
- ep
- cgp
- gpy
---

# Task: CIP-0033 Phase B — `CGp` FTC + EP

## Description

Wire `CEpInference` into dense `CGp` so hyperparameter optimisation uses the GPy-style EP approximate marginal (Bernoulli + Probit + RBF, no auto White). Add a GPy-generated or shared-numpy fixture for a tiny classification problem. Thin façade / CLI can follow once the C++ objective is green.

## Acceptance Criteria

- [x] `CGp` FTC path can run EP with probit/Bernoulli labels (`setEpInference(true)`)
- [x] Kernel hypers move under SCG with finite grads (FD vs analytic in `testEp`)
- [x] Fixture vs frozen numpy twin (`cip/cip0033/ep_ftc_fixture.txt`) within 1e-10
- [x] Tutorial note: [cgp_ep_classification.md](../../docs/tutorials/cgp_ep_classification.md); IVM cross-link updated

## Related

- CIP: 0033
- REQ: 0019

## Progress Updates

### 2026-11-21

Task created after Phase A `testEp` green.

### 2026-11-21 (later)

Phase B landed: `epLogLikelihood` / `updateG` EP branch / `invalidateCache` clears `epUpToDate`; makefile links `CEpInference` into `testEp` / `testCGp` / `gp` / `libgp`; fixture + tutorial.
