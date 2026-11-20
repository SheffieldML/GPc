---
id: "2026-11-20_cip000A-phase3-remaining-sites"
title: "CIP-000A Phase 3: Take-ownership factories + Reference green"
status: "Completed"
priority: "Medium"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip000A-phase2-cmpnd-tensor"]
tags:
- backlog
- kern
- memory
- matlab
- stream
---

# Task: CIP-000A Phase 3 remaining owning sites

> Backlog task for [CIP-000A](../../cip/cip000A.md) Phase 3.

## Description

Route stream and MATLAB component factories through `addKern(unique_ptr)` so allocated kernels are not cloned-and-leaked. Confirm Reference suite green. Optionally inventory remaining model-level owning `CKern*` (`CGp::pkern`, etc.) for a follow-on CIP if needed.

## Acceptance Criteria

- [x] `readParamsFromStream` uses take-ownership `addKern`
- [x] MATLAB `extractParamFromMxArray` uses take-ownership `addKern` (when `_NDLMATLAB`)
- [x] Full matio Reference suite green on this branch
- [ ] Ownership note at CIP close (compression)

## Implementation Notes

- `addKern(std::unique_ptr<CKern>(readKernFromStream(in)))` or equivalent.
- Prefer `std::make_unique` for MATLAB type switches.

## Related

- CIP: 000A
- Requirement: REQ-0004

## Progress Updates

### 2026-11-20

Task created Ready; depends on Phase 2.

### 2026-11-20 (partial)

Stream + MATLAB take-ownership landed with Phase 2. Remaining: full `USE_MATIO=1 matio-tests` and close-time docs note.

### 2026-11-20 (done)

`make run-matio-tests USE_MATIO=1 CHECKS=full` green (all failure counts 0). Docs ownership note deferred to CIP close. Status → Completed.
