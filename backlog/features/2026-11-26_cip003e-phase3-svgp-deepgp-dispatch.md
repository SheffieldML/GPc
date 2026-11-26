---
id: "2026-11-26_cip003e-phase3-svgp-deepgp-dispatch"
title: "Phase 3: SVGP and deep-GP dispatch through the block oracle"
status: "Ready"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003E"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- likelihood
- multiclass
---

# Task: Phase 3: SVGP and deep-GP dispatch through the block oracle

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

CSvgp data term, CVI natgrad sites and elboGrad loop over blocks; CDeepGp final layer wiring.

## Acceptance Criteria

- [ ] testSvgp, testDeepGp, testSviBayesGplvm unchanged at blockSize 1
- [ ] checkGradients passes with a block likelihood

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.
