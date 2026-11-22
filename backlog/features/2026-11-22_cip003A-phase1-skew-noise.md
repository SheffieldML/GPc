---
id: "2026-11-22_cip003A-phase1-skew-noise"
title: "CIP-003A Phase 1: SkewGaussian and SkewExponential CNoise"
status: "Completed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["003A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-22_cip003A-phase0-gaussian-special"]
tags:
- backlog
- likelihood
- skew
- noise
---

# Task: CIP-003A Phase 1 — Skew CNoise

## Description

Implement `CSkewGaussianNoise` and `CSkewExponentialNoise` (stream types `skew_gaussian` / `skew_exponential`), register in `readNoiseFromStream`, density/Laplace oracles in `ndlnoise`, and `testSkewNoise` (SciPy / alpha=0≡Gaussian / `checkGradients`).

## Acceptance Criteria

- [x] Both classes implement the `CNoise` contract needed for param optimisation and predictive moments
- [x] Stream round-trip via factory
- [x] `./testSkewNoise` green without matio
- [x] SkewGaussian `alpha=0` matches Gaussian log-density within tolerance

## Related

- CIP: 003A
- REQ: 001B

## Progress Updates

### 2026-11-22
Completed: `CSkewGaussianNoise` / `CSkewExponentialNoise`; factory types `skew_gaussian` / `skew_exponential`; `./testSkewNoise` all green (SciPy dens, FD grads, stream).
