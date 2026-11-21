---
id: "2026-11-21_cip0033-phase-b-cgp-ep"
title: "CIP-0033 Phase B: wire EP into CGp FTC + GPy fixture"
status: "Ready"
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

- [ ] `CGp` FTC path can run EP with probit/Bernoulli labels
- [ ] Kernel hypers move under SCG (or existing optimiser) with finite grads
- [ ] Fixture vs GPy `GPClassification` (or frozen numpy twin) within documented tol
- [ ] Tutorial note: EP path vs IVM substitute

## Related

- CIP: 0033
- REQ: 0019

## Progress Updates

### 2026-11-21

Task created after Phase A `testEp` green.
