---
id: "2026-11-25_cip0016-phase-e-python-ml-stacks"
title: "CIP-0016: Phase E: GPyTorch, GPflow, GPJax adapters"
status: "Ready"
priority: "Medium"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["0016"]
owner: "Neil Lawrence"
dependencies: ["2026-11-25_cip0016-phase-c-gpy-roundtrip"]
tags:
- backlog
- interop
- gp-scuderia
---

# Task: CIP-0016 Phase E: GPyTorch, GPflow, GPJax adapters

## Description

Verify the provisional registry entries against current releases of GPyTorch, GPflow and GPJax; write adapters; round-trip the goldens; confirm supported approximations (currently provisional: ftc and dtcvar only); GPyTorch must reject kern.white; GPJax uses sqrt for obs_stddev; document gaps.

## Acceptance Criteria

- [ ] See Description; verified against goldens in `gp-scuderia`

## Related

- CIP: 0016
- Requirement: 000A

## Progress Updates

### 2026-11-25
Task created with Ready status on acceptance of CIP-0016.
