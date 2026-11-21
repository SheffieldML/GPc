---
id: "2026-11-21_cip0033-phase-d-surfaces"
title: "CIP-0033 Phase D: site workspace, classify façade, FTC-only docs"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0033"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0033-phase-c-cgp-laplace"]
tags:
- backlog
- classification
- facade
- noise
---

# Task: CIP-0033 Phase D — strip / surfaces

## Acceptance Criteria

- [x] `CNoiseSiteWorkspace` dual-write; PPA attach proof; `testEp` coverage
- [x] FTC-only / no sparse EP documented (`phase_d_surfaces.md`)
- [x] Stream note: params only
- [x] `gpc.GPClassification` + Python tests
- [ ] Full removal of `mu`/`varSigma` from `CNoise` (follow-on; REQ-0019 first box)

## Related

- CIP: 0033
- REQ: 0019
