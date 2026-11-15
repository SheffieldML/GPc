---
id: "2026-11-15_cgp-coverage"
title: "Near-complete CGp.cpp coverage (CIP-0011)"
status: "Completed"
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

Implement [CIP-0011](../../cip/cip0011.md): baseline lcov for `CGp.cpp`, classify misses (cover / expect-throw / delete), add `testCGp`, and enforce a file-level line-rate gate (≥95%) in the Coverage workflow.

## Acceptance Criteria

- [x] Appendix A miss list filled from Coverage/`COVERAGE=1` lcov
- [x] Coverable gaps have tests; throw stubs covered or unreachable tails removed
- [x] CI asserts `CGp.cpp` line rate ≥ 95% (`scripts/check_cgp_coverage.sh`)
- [x] CIP-0011 Implemented after local green
- [ ] CIP-0011 Closed after green Coverage CI

## Related

- CIP: 0011
- REQ: 0002
- PR: https://github.com/SheffieldML/GPc/pull/37

## Progress Updates

### 2026-11-15

Task created with CIP-0011 Proposed on main (after CIP-000E/000F merged).

### 2026-11-15 (later)

Accepted and implemented on `cip0011-cgp-coverage`: baseline 71% → ~96% with `testCGp`; unreachable non-spherical LL/grad tails removed; Coverage workflow gate at 95%. PR #37 opened.
