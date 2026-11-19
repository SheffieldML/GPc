---
id: "2026-11-19_cip0028-phase-1b-mlp-ard"
title: "CIP-0028 Phase 1b: MLP ARD weight-space path"
status: "Completed"
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

- [x] `CMlpardKern` implements `wsCross*` / grads
- [x] FD gates for ARD weight/bias variances and centres
- [x] Sparse DTCVAR smoke with MLP-ARD + weight-space

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-1-mlp-isotropic

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted.

### 2026-11-19 (done)
Diagonal \(\Sigma=\mathrm{diag}(\alpha s_j,\,\beta)\) with isotropic window \(s\); scales=1 matches isotropic; DTCVAR 2-D smoke + FD on inputScale green.

### 2026-11-19 (ridge revision)
ARD under ridge falls out of \(\hat{\mathbf x}=\Sigma^{1/2}\tilde{\mathbf x}\); re-gate after [`2026-11-19_cip0028-phase-1-ridge`](./2026-11-19_cip0028-phase-1-ridge.md). Isotropic ARD path is interim only.
