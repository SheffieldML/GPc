---
id: "2026-11-15_test-optimisable"
title: "Implement testOptimisable suite (CIP-0010)"
status: "Completed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["0010"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- testing
- optimiser
- infrastructure
---

# Task: testOptimisable suite

## Description

Ship `testOptimisable` with a quadratic toy `COptimisable`, cover gradients and SCG/CG/GD/L-BFGS plus optimiser string synonyms, and wire it into the reference run gate.

## Acceptance Criteria

- [x] `make testOptimisable && ./testOptimisable` exits 0
- [x] Included in `matio-tests` / `run-matio-tests`
- [x] Documented in `docs/build-and-ci.md`
- [x] CIP-0010 marked Implemented after local green
- [ ] CIP-0010 Closed after PR CI

## Related

- CIP: 0010
- REQ: 0002

## Progress Updates

### 2026-11-15

Task created; implementation on `cip0010-test-optimisable`. Local `./testOptimisable` green (0 failures). CIP-0010 → Implemented; close after PR CI.
