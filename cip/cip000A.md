---
author: "Neil Lawrence"
created: "2026-11-13"
id: "000A"
last_updated: "2026-11-20"
status: "Implemented"
compressed: false
related_requirements: ["0004"]
related_cips: ["0003", "0008", "0009"]
tags:
- cip
- kern
- memory
- modernization
- infrastructure
title: "Own compound kernels with unique_ptr"
---

# CIP-000A: Own compound kernels with unique_ptr

## Status

- [x] Proposed
- [x] Accepted
- [x] In Progress - worktree `cip000A-unique-ptr-kern` / branch `cip000A-unique-ptr-kern`
- [x] Implemented
- [ ] Closed

## Backlog

- [Phase 1 — inventory](../backlog/features/2026-11-20_cip000A-phase1-inventory.md) (High)
- [Phase 2 — migrate CCmpndKern / CTensorKern](../backlog/features/2026-11-20_cip000A-phase2-cmpnd-tensor.md) (High)
- [Phase 3 — remaining owning sites + Reference green](../backlog/features/2026-11-20_cip000A-phase3-remaining-sites.md) (Medium)

## Summary

Replace manual `new` / `delete` graphs for component kernels (especially in `CCmpndKern` and `CTensorKern`) with `std::unique_ptr` (or an equivalent clear ownership type) so copy/slash constructors cannot leak or double-free, and so the CIP-0003 tensor bug class is structurally harder to repeat.

## Motivation

`CKern.cpp` alone contains on the order of forty `new` / `delete` pairs for component kernels. Compound and tensor kernels own vectors of `CKern*`. The tensor slash/copy bug fixed in [CIP-0003](cip0003.md) was exactly this class of error: shallow copy then re-`addKern` corrupted leave-one-out gradient products. Manual ownership in a 4,400-line translation unit makes similar bugs likely.

This continues [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md) beyond language mode into ownership. Prefer doing this **after** [CIP-0008](cip0008.md) so matrix values used beside kernels are already assignment-safe. (CIP-0008 is Closed.)

## Detailed Description

### Current state

- Component lists are raw pointers with hand-written destructors and copy paths.
- Slash constructors allocate temporary kernels for gradient products.
- Public APIs take and return `CKern*` in many places.
- `CKern::clone()` already exists; `CComponentKern::addKern(const CKern*)` already clones into the component list.

### Ownership inventory (Phase 1)

| Site | Role | Notes |
|------|------|--------|
| `CComponentKern::components` | **Owning** | Today `vector<CKern*>`; destructor of `CCmpndKern`/`CTensorKern` deletes each. Target: `vector<unique_ptr<CKern>>`. |
| `addKern(const CKern*)` | Observer in → clone owned | Caller retains the argument; clone is owned by `components`. |
| `addKern(unique_ptr<CKern>)` (new) | **Owning transfer** | No clone; used for stream/MATLAB factories that allocate. Fixes pre-existing leak where `addKern(new …)` / `addKern(readKernFromStream)` cloned then orphaned the temporary. |
| `CGp::pkern` / noise | **Owning or borrowed** | Out of scope for Phase 2 unless tests force it; inventory only. |
| `CKern*` args to `compute` / `getGrad*` | Non-owning | Unchanged. |
| Slash `CTensorKern(const CTensorKern&, i)` | Deep-copies via `addKern` | Must keep cloning leave-one-out children into owned `unique_ptr`s. |
| `CComponentKern(const CComponentKern&)` shallow copy | **Bug class** | Shallow-copied the pointer vector. With `unique_ptr`, delete this ctor (move-only components); subclasses deep-copy via empty base + `addKern`. |

### Approach

1. **Inventory** — done above; refine if Phase 3 finds more owning sites.
2. **Owning containers** — `std::vector<std::unique_ptr<CKern>>`; `addKern(const CKern*)` clones; `addKern(unique_ptr)` moves.
3. **Copy / clone** — keep virtual `clone()`; delete shallow `CComponentKern` copy; compound/tensor copy and slash paths use `addKern`.
4. **API boundary** — transitional non-owning `const CKern*` `addKern` remains; document ownership in header comments.
5. **Split** — defer splitting `CKern.cpp` to a later CIP.

### Non-goals

- Rewriting every kernel algorithm.
- Switching the public C API / SWIG surface in one step (note bindings impact if any remain).
- `shared_ptr` unless an inventory item proves shared ownership is real (prefer unique ownership).

## Implementation Plan

1. Inventory owning vs observing pointers; record in this CIP. ✅
2. Migrate `CCmpndKern` and `CTensorKern` first (highest bug history).
3. Fix slash/copy/clone paths; re-run tensor and compound kern tests.
4. Migrate remaining owning sites as needed for a green Reference suite.
5. Document ownership rules in `docs/` when closed.

## Implementation Status

- [x] Accepted; backlog tasks created
- [x] Phase 1 inventory written into this CIP
- [x] Phase 2: `unique_ptr` components + `addKern` overloads; remove manual deletes
- [x] Phase 3: stream/MATLAB take-ownership paths (done with Phase 2)
- [x] Full matio Reference suite green (`USE_MATIO=1 CHECKS=full run-matio-tests`)
- [ ] Remaining model-level owning sites (`CGp::pkern`, etc.) — deferred (out of scope)
- [ ] Close + docs ownership note

## Testing Strategy

- Existing `testKern` compound and tensor cases (regression for CIP-0003).
- Full matio Reference suite.
- Compile under C++17 (already the project floor).

## Related

- [CIP-0003](cip0003.md) tensor slash/copy fix
- [CIP-0008](cip0008.md) `CMatrix` ownership (Closed — prerequisite)
- [CIP-0009](cip0009.md) GP caches (Closed — deferred kernel ownership here)
- Review note: manual kernel ownership (2026-11-13)
