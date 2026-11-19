---
id: "2026-11-19_cip0029-phase-0-skeleton"
title: "CIP-0029 Phase 0: CInducingGreens skeleton + CLI/stream"
status: "Ready"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0029"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- inducing
- greens
- causal
---

# Task: CIP-0029 Phase 0 — `CInducingGreens` skeleton

## Description

Add `CInducingGreens` as a fourth inducing family (`typeName()=="greens"`), with kernel-free Gaussian-window \(K_{uu}\), 1-D centres on existing `X_u`, CLI `--inducing greens`, stream keys, and `computeKuf` / greens cross-hooks that throw until Phase 1. Assert input dimension is 1 when this family is selected. Share window-Gram helpers with white-noise inducing where practical; do not change `whitenoise` behaviour.

## Acceptance Criteria

- [ ] `CInducingGreens` implements `CInducing` with `kuuDependsOnKernel()==false`, `domainDim()==1`, fixed window width \(s\)
- [ ] `CGp::setInducingType("greens")` and `gp --inducing greens` (+ `--window-width`) work; non-1-D inputs error clearly
- [ ] Stream round-trip writes/reads `inducingType=greens` and `windowWidth`
- [ ] `computeKuf` / prediction path throws a clear `NotImplementedError` (or equivalent) until Phase 1 kernels land
- [ ] Existing point and white-noise tests remain green

## Implementation Notes

Centres stay in input space via `X_u` (unlike CIP-0028 weight-space). Reuse median-spacing default for \(s\) from `CInducingWhiteNoise` where appropriate.

## Related

- CIP: 0029
- REQ: 000D

## Progress Updates

### 2026-11-19
Task created (Ready) after CIP-0029 Accepted.
