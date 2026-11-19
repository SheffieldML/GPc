---
id: "2026-11-19_cip0028-phase-1b-mlp-ard"
title: "CIP-0028 Phase 1b: MLP ARD weight-space path"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0028-phase-1-mlp-isotropic"]
tags:
- backlog
- sparse
- inducing
- weight-space
- mlp
- ard
---

# Task: CIP-0028 Phase 1b — MLP ARD weight-space

## Description

Extend weight-space cross-covariance to `CMlpardKern` (`wsCross*` overrides). Start with isotropic window width \(s\) in weight space even when \(\Sigma\) is diagonal ARD; optional per-coordinate window metric later if needed.

## Acceptance Criteria

- [ ] `CMlpardKern` implements `wsCross*` / grads
- [ ] FD gates for ARD weight/bias variances and centres
- [ ] Sparse DTCVAR smoke with MLP-ARD + weight-space

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-1-mlp-isotropic

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted.
