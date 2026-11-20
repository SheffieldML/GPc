---
id: "2026-11-20_cip0027-phase-a-likelihood-variance"
title: "CIP-0027 Phase A: likelihood-owned σ² + warnings + FTC equivalence"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0027"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0027-phase0-likelihood-api"]
tags:
- backlog
- noise
- gaussian
- ftc
- deprecation
---

# Task: CIP-0027 Phase A — Likelihood-owned σ²

## Description

Every Gaussian-observation model takes σ² from the likelihood (`likelihood.variance`, alias `Gaussian_noise.variance`). FTC adds σ² on the Gram after `pkern->compute`. Warn if trainable White is also present while Gaussian variance is free. Legacy White-only noise models keep working under a documented fallback. Equivalence gate: FTC RBF + `likelihood.variance` ≡ legacy RBF+White.

## Acceptance Criteria

- [x] Canonical path `likelihood.variance` (+ GPy1 alias) get/set on Gaussian path
- [x] FTC Gram includes σ² from likelihood; gradients via `checkGradients`
- [x] Sparse models sync β ↔ σ² where algebra needs precision
- [x] Deprecation warning when White + free likelihood variance both trainable
- [x] `testNoiseInLikelihood` (or equivalent) passes within `MATCHTOL`

## Related

- CIP: 0027
- Depends on: Phase 0

## Progress Updates

### 2026-11-20

Task created on Accept.

### 2026-11-20 (done)

FTC Gram + predictive σ²; path aliases; sparse β sync; dual-noise warn; legacy White skip fallback; `testNoiseInLikelihood` + `testCGp` green.
