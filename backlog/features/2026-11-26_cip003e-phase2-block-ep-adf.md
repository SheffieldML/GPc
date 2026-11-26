---
id: "2026-11-26_cip003e-phase2-block-ep-adf"
title: "Phase 2: block EP (N x K sites), ADF in IVM/PPA/sparse PPA/CGp, capability refusals"
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

# Task: Phase 2: block EP (N x K sites), ADF in IVM/PPA/sparse PPA/CGp, capability refusals

> Implements CIP-003E (HOW). See the CIP for the design and verification plan.

## Description

Generalise CEpInference to N x K diagonal sites via cavityMomentsBlock; ADF engines use the same oracle; IVM selection sums entropy change over the block; refuse likelihoods lacking capabilities.

## Acceptance Criteria

- [ ] CEpInference N x K sites; K = 1 reproduces testEp fixtures exactly
- [ ] ADF in CIvm/CPpa/CSparsePpa/CGp through cavityMomentsBlock
- [ ] IVM block selection reduces to current rule at K = 1
- [ ] Capability refusal tests (EP/ADF without CapCavityMoments, Laplace without CapPointDerivs)

## Implementation Notes

Work on branch `cip003E-multiclass-likelihood` (worktree `../GPc-cip003E-multiclass`). Phases are ordered; phase 1 gates the rest.

## Related

- CIP: 003E

## Progress Updates

### 2026-11-26
Task created with Ready status when CIP-003E was accepted.
