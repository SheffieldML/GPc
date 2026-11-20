---
id: "2026-11-20_cip0027-phase-b-default-flip"
title: "CIP-0027 Phase B: flip gp/ppa/demo/Python defaults (no auto White/Bias)"
status: "Ready"
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

- [ ] Regression/PPA/demo/Python defaults do not auto-append White or Bias
- [ ] Explicit `+ White` / `+ Bias` still works
- [ ] Docs / release notes describe the user-visible break
- [ ] CIP-0026 White carve-out notes updated
- [ ] Smoke: Python/demo regression without Bias/White

## Related

- CIP: 0027, 001D
- Depends on: Phase A

## Progress Updates

### 2026-11-20

Task created on Accept.
