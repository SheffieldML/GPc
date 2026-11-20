---
id: "2026-11-20_cip000A-phase2-cmpnd-tensor"
title: "CIP-000A Phase 2: unique_ptr components for CCmpndKern / CTensorKern"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip000A-phase1-inventory"]
tags:
- backlog
- kern
- memory
- unique_ptr
---

# Task: CIP-000A Phase 2 migrate compound/tensor ownership

> Backlog task for [CIP-000A](../../cip/cip000A.md) Phase 2.

## Description

Change `CComponentKern::components` to `vector<unique_ptr<CKern>>`. Delete the shallow `CComponentKern` copy constructor. Keep `addKern(const CKern*)` as clone-in; add `addKern(unique_ptr<CKern>)` for ownership transfer. Default compound/tensor destructors. Fix copy and slash constructors to pass `.get()` into cloning `addKern`.

## Acceptance Criteria

- [x] `components` is `vector<unique_ptr<CKern>>`
- [x] Shallow `CComponentKern` copy ctor deleted
- [x] Manual `delete` loops removed from `~CCmpndKern` / `~CTensorKern`
- [x] Copy and slash paths still deep-copy via `addKern`
- [x] `testKern` compound and tensor cases pass

## Implementation Notes

- `#include <memory>` in `CKern.h`.
- `CTensorKern::addKern` override must accept both raw (clone) and `unique_ptr` (transfer) for white/indexwhite checks.

## Related

- CIP: 000A
- Requirement: REQ-0004

## Progress Updates

### 2026-11-20

Task created; implementation underway on `cip000A-unique-ptr-kern`.

### 2026-11-20 (done)

`unique_ptr` components + take-ownership `addKern`; `testKern` 0 failures (compound + tensor included). Status → Completed.
