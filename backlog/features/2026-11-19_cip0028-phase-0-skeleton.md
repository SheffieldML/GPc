---
id: "2026-11-19_cip0028-phase-0-skeleton"
title: "CIP-0028 Phase 0: CInducingWeightSpace skeleton + CLI/stream"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- inducing
- weight-space
- mlp
---

# Task: CIP-0028 Phase 0 — `CInducingWeightSpace` skeleton

## Description

Add `CInducingWeightSpace` as a third/fourth inducing family (`typeName()=="weightspace"`), Design A: owns `W` (`M × (d+1)`), `centres()` returns `W`, `X_u` unused for this type, kernel-free Gaussian-window \(K_{uu}\) in weight dim \(d+1\), CLI `--inducing weightspace`, stream keys, and `computeKuf` / ws\* cross-hooks that throw until Phase 1.

## Acceptance Criteria

- [x] `CInducingWeightSpace` implements `CInducing` with `kuuDependsOnKernel()==false`, `domainDim()==inputDim+1`, fixed window width \(s\)
- [x] `CGp::setInducingType("weightspace")` and `gp --inducing weightspace` (+ `--window-width`) work
- [x] Stream round-trip writes/reads `inducingType=weightspace`, `windowWidth`, and `weightCentres`
- [x] `computeKuf` / prediction path throws a clear `NotImplementedError` until Phase 1 MLP hooks land
- [x] Existing point, white-noise, and greens tests remain green

## Implementation Notes

Prefer Design A from CIP-0028 (inducing owns `W`; do not resize `X_u` to \(d+1\)). Reuse median-spacing default for \(s\) from `CInducingWhiteNoise::medianSpacing(W)`.

## Related

- CIP: 0028
- REQ: 000D

## Progress Updates

### 2026-11-19
Task created (Ready) after CIP-0028 Accepted.

### 2026-11-19 (done)
`CInducingWeightSpace` + `setInducingType("weightspace")` / stream / CLI help; `wsCross*` defaults throw; `testWeightSpaceInducingPhase0` green (`testCGp` failures=0).

