---
id: "2026-11-21_cip0031-phase-b-models"
title: "CIP-0031 Phase B: latent/predictive on remaining models"
status: "Completed"
priority: "Medium"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0031"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0031-phase-a-cgp"]
tags:
- backlog
- prediction
- ppa
- svgp
- gplvm
---

# Task: CIP-0031 Phase B — Other model classes

## Description

Add `latentMeanVar` / `predictiveMeanVar` on `CPpa`, `CSparsePpa`, `CIvm`, `CBayesGplvm`, `CMrd`, `CSvgp` (and `CSviBayesGplvm` if it exposes the same entry). Document dense↔sparse PPA agreement at \(M=N\).

## Acceptance Criteria

- [x] Pair exists on each scoped class with documented semantics
- [x] Dense vs sparse PPA predictive input agree at \(M=N\) (no White); semantics documented
- [x] `CSvgp::latentMeanVar` matches `CGp::latentMeanVar` at matched DTCVAR without manual \(1/\beta\) correction
- [x] Tests in the relevant binaries

## Related

- CIP: 0031
- Depends on: Phase A

## Progress Updates

### 2026-11-21

Task created Ready on CIP Accept.

### 2026-11-21 (complete)

API + `out` latent routing landed. Alias table: SVGP/dense PPA/IVM/MAP GPLVM `posteriorMeanVar`→latent; Bayes GPLVM/sparse PPA→predictive. Gates in `testSvgp` and `testSparsePpa`.
