---
id: "2026-11-20_cip0027-phase-b-default-flip"
title: "CIP-0027 Phase B: flip gp/ppa/demo/Python defaults (no auto White/Bias)"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0027", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0027-phase-a-likelihood-variance"]
tags:
- backlog
- noise
- white
- bias
- cli
- python
---

# Task: CIP-0027 Phase B — Default flip (regression / PPA / demos / Python)

## Description

Stop auto-`addKern` of White and Bias in `gp.cpp`, `ppa.cpp`, `demo_regression_1d`, and Python `GPRegression`. Default compound = user-requested kernels only. Update tutorials / CIP-001D docs. Update CIP-0026 FTC/White notes. **Do not flip `ivm.cpp`.**

## Acceptance Criteria

- [x] Regression/PPA/demo/Python defaults do not auto-append White or Bias
- [x] Explicit `+ White` / `+ Bias` still works
- [x] Docs / release notes describe the user-visible break
- [x] CIP-0026 White carve-out notes updated
- [x] Smoke: Python/demo regression without Bias/White

## Related

- CIP: 0027, 001D
- Depends on: Phase A

## Progress Updates

### 2026-11-20

Task created on Accept.

### 2026-11-20 (done)

Removed auto Bias/White from `gp.cpp`, `ppa.cpp`, demos, Python `addFtcStyleCompound`; FTC default `likelihood.variance=1`; docs + CIP-0026 notes; IVM unchanged.
