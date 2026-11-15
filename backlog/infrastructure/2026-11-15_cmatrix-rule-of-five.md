---
id: "2026-11-15_cmatrix-rule-of-five"
title: "Implement CMatrix Rule of Five ownership (CIP-0008)"
status: "Completed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["0008"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- matrix
- memory
- cxx
- infrastructure
---

# Task: CMatrix Rule of Five ownership

## Description

Close the Rule of Three gap on `CMatrix`: add deep copy assignment plus move constructor/assignment so `A = B` cannot alias `vals` into a double-free. Prefer Rule of Five over deleted assignment (inventory shows return-by-value helpers and few dangerous assign sites; move benefits those paths).

## Acceptance Criteria

- [x] Copy and move assignment implemented; self-assignment safe
- [x] Moved-from state is destructible and assignable (`vals == NULL` allowed in `memDeAllocate`)
- [x] `testOwnership` in `testMatrix` covers copy-construct, assign, self-assign, move, use-after-assign independence
- [x] `make USE_MATIO=1 testMatrix` green locally (0 failures)
- [ ] Compile / Reference CI green on PR
- [x] Ownership rule noted in `docs/numerics.md`; CIP-0008 decision recorded

## Implementation Notes

- Inventory: value init from `chol`/`inv`/`multiply`/`mean*`/`sum*`/`var*` (copy/move elision); explicit assigns rare — codebase prefers `deepCopy`. Containers hold `CMatrix*`, not `CMatrix` by value.
- Decision: **Rule of Five** (not `= delete` assignment).

## Related

- CIP: 0008
- REQ: 0004

## Progress Updates

### 2026-11-15

Task created with CIP-0008 Accepted → In Progress on `cip0008-cmatrix-ownership`.

Implemented Rule of Five on `CMatrix`; `testOwnership` passes; docs updated. Awaiting PR CI before closing CIP-0008.
