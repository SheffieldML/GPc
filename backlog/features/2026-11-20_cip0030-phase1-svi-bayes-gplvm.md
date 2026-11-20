---
id: "2026-11-20_cip0030-phase1-svi-bayes-gplvm"
title: "CIP-0030 Phase 1: CSviBayesGplvm Gaussian points/RBF"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "002A", "0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0030-phase0b-explicit-input-core"]
tags:
- backlog
- svgp
- svi
- bayesian-gplvm
- reparameterisation
---

# Task: CIP-0030 Phase 1 — `CSviBayesGplvm` (Gaussian; points / RBF)

## Description

Implement uncollapsed Bayesian GP-LVM via reparameterised \(q(\mathbf X)=\prod_n\mathcal N(\boldsymbol\mu_n,\mathrm{diag}\,\mathbf S_n)\) on top of `CSvgp`'s explicit-input core. No psi statistics. Stochastic params for `CAdam` are globals (kern, inducing, \(\beta\)); local \((\boldsymbol\mu,\log\mathbf S)\) use row-sparse Adam on the minibatch. CVI natgrad on \(q(\mathbf v)\) via `afterBatchStep` → `naturalGradientStepAt`.

## Acceptance Criteria

- [x] `CSviBayesGplvm` owns \(\mathbf Y\), \(\boldsymbol\mu\), \(\log\mathbf S\), seeded sampler, \(K\), and a `CSvgp` over points / RBF(+White)
- [x] `elboEstimate` / `objectiveGradBatch` stack \(B\times K\) samples; KL\((q(\mathbf u))\) and kernel prior counted once; \(-\mathrm{KL}(q(\mathbf X))\) scaled by \(N/B\)
- [x] Row-sparse Adam on \((\boldsymbol\mu,\log\mathbf S)\) (disable with `lr=0` for oracles / FD)
- [x] `testSviBayesGplvm` gates V1–V3, V5–V7 (psi oracles for V2/V3; FD for V5; partition V6; seed V7)
- [x] Wired into makefile `tests` list

## Implementation Notes

Prefer collapsed `CBayesGplvm` when psi exist; this path is for inducing families without psi and for the deep-GP seam (KL / \(\boldsymbol\mu,\log\mathbf S\) stay out of `CSvgp`).

## Related

- CIP: 0030
- Depends: Phase 0b explicit-input core

## Progress Updates

### 2026-11-20

Implemented `CSviBayesGplvm.{h,cpp}` and `testSviBayesGplvm.cpp`. All Phase 1 gates green (`Number of failures: 0`).
