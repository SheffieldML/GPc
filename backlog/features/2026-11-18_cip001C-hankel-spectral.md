---
id: "2026-11-18_cip001C-hankel-spectral"
title: "CIP-001C Phase 3: spectralDensity + Hankel path for Matérn/Exp"
status: "Ready"
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

- [ ] `spectralDensity(rho)` on Matérn 3/2, 5/2, Exp
- [ ] Hankel tabulation + interpolation for \(K_{fu}\) and lengthscale/variance grads
- [ ] RBF-via-Hankel matches closed form to `MATCHTOL`
- [ ] Matérn matches 1-D/2-D numerical convolution on a small grid

## Related

- CIP: 001C
- REQ: 000D
