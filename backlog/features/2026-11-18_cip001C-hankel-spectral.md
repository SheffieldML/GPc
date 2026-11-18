---
id: "2026-11-18_cip001C-hankel-spectral"
title: "CIP-001C Phase 3: spectralDensity + Hankel path for Matérn/Exp"
status: "Completed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001C"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- inducing
- spectral
- matern
---

# Task: CIP-001C Phase 3 — Hankel / spectralDensity path

## Description

Extend white-noise inducing beyond closed-form RBF to isotropic stationary kernels via `CStationaryKern::spectralDensity` and a 1-D Hankel transform for \(K_{fu}\). Gate by requiring the generic path on RBF to match Phase 2 closed form.

## Acceptance Criteria

- [x] `spectralDensity(rho)` on Matérn 3/2, 5/2, Exp
- [x] Hankel tabulation + interpolation for \(K_{fu}\) and lengthscale/variance grads
- [x] RBF-via-Hankel matches closed form to `MATCHTOL` (gate: max rel err \(\sim 10^{-10}\) on `testWnHankel`)
- [x] Matérn matches 1-D/2-D numerical convolution on a small grid (1-D cosine IFT of \(\sqrt S\hat\psi\); Exp spectral matches Matérn-\(\nu=1/2\))

## Related

- CIP: 001C
- REQ: 000D

## Progress Updates

### 2026-11-18
Implemented on `cip001C-wn-pilot`: `spectralDensity` / `spectralLengthScale`, `wnCrossComputeHankel`, stationary default `wnCross*` (FD grads), `testWnHankel` makefile target. RBF closed-form path unchanged.
