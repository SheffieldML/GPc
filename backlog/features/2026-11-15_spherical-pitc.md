---
id: "2026-11-15_spherical-pitc"
title: "Spherical PITC + matio fixture (CIP-000F)"
status: "Completed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "features"
related_cips: ["000F"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- pitc
- sparse
- testing
---

# Task: Spherical PITC

## Acceptance Criteria

- [x] Spherical PITC LL/grads; `diagPitc` PASS
- [x] `matfiles/testGppitc.mat` + `testGaussian("pitc")`
- [x] Docs note; CIP-000F Implemented
- [ ] Closed after PR CI

## Progress Updates

### 2026-11-15

Implemented on `cip000F-pitc`. Local `diagPitc` and `testGp` (incl. pitc) green.
