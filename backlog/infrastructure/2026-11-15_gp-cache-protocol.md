---
id: "2026-11-15_gp-cache-protocol"
title: "Explicit GP/GPLVM cache invalidation protocol (CIP-0009)"
status: "Completed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["0009"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- gp
- gplvm
- caching
- modernization
- infrastructure
---

# Task: Explicit GP / GPLVM cache protocol

## Description

Accept and implement CIP-0009: document the mutable working-state inventory, expose `invalidateCache` / `ensure*UpToDate` on `CGp` and `CGplvm`, wire mutators through invalidation, and add a regression that asserts flag behaviour after `setOptParams`.

## Acceptance Criteria

- [x] Protocol API on `CGp` / `CGplvm`; inventory recorded in CIP-0009
- [x] `setOptParams` / `updateX` / approximation-type changes invalidate correctly
- [x] `testGp` cache-invalidation case green (plus existing matio arms)
- [x] Maintainer note in `docs/numerics.md`
- [x] CIP-0009 Implemented after local green
- [x] CIP-0009 Closed after PR CI

## Related

- CIP: 0009
- REQ: 0002, 0004
- PR: https://github.com/SheffieldML/GPc/pull/32

## Progress Updates

### 2026-11-15

Task created on `cip0009-gp-caches`. Local `./testGp` green (cache case + ftc/dtc/dtcvar/fitc). CIP-0009 → Implemented; PR #32 CI green → Closed.
