---
id: "2026-11-15_cgp-coverage"
title: "Near-complete CGp.cpp coverage (CIP-0011)"
status: "Proposed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["0011"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- coverage
- gp
- testing
- infrastructure
---

# Task: Near-complete CGp.cpp coverage

## Description

Implement [CIP-0011](../../cip/cip0011.md): baseline lcov for `CGp.cpp`, classify misses (cover / expect-throw / delete), add drivers, and enforce a file-level line-rate gate in the Coverage workflow.

## Acceptance Criteria

- [ ] Appendix A miss list filled from Coverage/`COVERAGE=1` lcov
- [ ] Coverable gaps have tests; throw stubs covered or removed (CIP-000B)
- [ ] CI asserts `CGp.cpp` line rate ≥ documented threshold (95% → 98%)
- [ ] CIP-0011 Implemented then Closed after green Coverage CI

## Related

- CIP: 0011
- REQ: 0002

## Progress Updates

### 2026-11-15

Task created with CIP-0011 Proposed on main (after CIP-000E/000F merged).
