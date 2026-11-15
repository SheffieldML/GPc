---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000F"
last_updated: "2026-11-15"
status: "Implemented"
compressed: false
related_requirements: ["0002"]
related_cips: ["000E"]
tags:
- cip
- sparse
- pitc
- gp
- verification
- testing
title: "Implement spherical PITC and ship a matio reference fixture"
---

# CIP-000F: Implement spherical PITC and ship a matio reference fixture

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Finish the partially independent training conditional (PITC) approximation on the spherical Gaussian path in `CGp`, then add `matfiles/testGppitc.mat` and enable it in the matio `testGp` gate — matching the DTCVAR fixture workflow.

## Implementation Status

- [x] Inventory PITC stubs and block-state gaps
- [x] Implement spherical PITC `updateAD` / LL / gradients
- [x] Default `blockEnd` (GPmat `gpCreate` compatible)
- [x] Add regenerator + `matfiles/testGppitc.mat`
- [x] Enable `pitc` in `testGp` / matio CI
- [x] `diagPitc` FD gate green locally
- [ ] CI green; CIP Closed

## Testing Strategy

- `make diagPitc && ./diagPitc`
- `make USE_MATIO=1 testGp && ./testGp` includes `pitc`

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md)
- Related: [CIP-000E](cip000E.md) (optional StableFactor follow-on for PITC)
