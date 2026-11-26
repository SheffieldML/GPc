---
id: "2026-11-26_cip003d-phase3-gradients"
title: "CIP-003D phase 3: kernel-parameter gradients"
status: "Ready"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_cip003d-phase2-kuu-kfu"]
tags:
- backlog
- deepgp
- weightspace
- hermite
---

# Task: CIP-003D phase 3: kernel-parameter gradients

## Description

Implement `gradFromKuuKuf` for kernel parameters only (windows are fixed, `gZ` empty, `getNumParams()==0`) and the \(\partial/\partial\hat x\) path for data-X gradients.

## Acceptance Criteria

- [ ] `checkGradients` agrees with finite differences on a sparse GP with Hermite inducing (variance, weight variance, bias variance, and ARD scales for `CMlpardKern`)
- [ ] `CSvgp` / `CDeepGp` layer gradients agree with finite differences
- [ ] Gradients that disagree do not ship: failing check blocks the phase

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.
