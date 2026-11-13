---
author: "Neil Lawrence"
created: "2026-11-13"
id: "000A"
last_updated: "2026-11-13"
status: "Proposed"
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
- [ ] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed

## Summary

Replace manual `new` / `delete` graphs for component kernels (especially in `CCmpndKern` and `CTensorKern`) with `std::unique_ptr` (or an equivalent clear ownership type) so copy/slash constructors cannot leak or double-free, and so the CIP-0003 tensor bug class is structurally harder to repeat.

## Motivation

`CKern.cpp` alone contains on the order of forty `new` / `delete` pairs for component kernels. Compound and tensor kernels own vectors of `CKern*`. The tensor slash/copy bug fixed in [CIP-0003](cip0003.md) was exactly this class of error: shallow copy then re-`addKern` corrupted leave-one-out gradient products. Manual ownership in a 4,400-line translation unit makes similar bugs likely.

This continues [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md) beyond language mode into ownership. Prefer doing this **after** [CIP-0008](cip0008.md) so matrix values used beside kernels are already assignment-safe.

## Detailed Description

### Current state

- Component lists are raw pointers with hand-written destructors and copy paths.
- Slash constructors allocate temporary kernels for gradient products.
- Public APIs take and return `CKern*` in many places.

### Approach

1. **Inventory** — Classify each `CKern*` as owning vs non-owning (observers passed into `compute`, etc.).
2. **Owning containers** — Store components as `std::vector<std::unique_ptr<CKern>>` (or `std::unique_ptr<CKern[]>` only if a measured need appears). Provide `addKern` overloads that take `std::unique_ptr<CKern>` or allocate internally.
3. **Copy / clone** — Define an explicit virtual `clone()` on `CKern` if deep copy of compound trees is required; do not rely on pointer copy. Slash paths should clone into `unique_ptr` locals.
4. **API boundary** — Keep a transitional raw-pointer accessor where callers are not yet updated, documented as non-owning. Prefer updating in-tree callers in the same CIP when the blast radius is small.
5. **Split (optional follow-on)** — If the ownership change already forces large diffs, defer splitting `CKern.cpp` by kernel family to a separate CIP; mention the split as desirable for reviewability.

### Non-goals

- Rewriting every kernel algorithm.
- Switching the public C API / SWIG surface in one step (note bindings impact if any remain).
- `shared_ptr` unless an inventory item proves shared ownership is real (prefer unique ownership).

## Implementation Plan

1. Inventory owning vs observing pointers; record in this CIP.
2. Migrate `CCmpndKern` and `CTensorKern` first (highest bug history).
3. Fix slash/copy/clone paths; re-run tensor and compound kern tests.
4. Migrate remaining owning sites as needed for a green Reference suite.
5. Document ownership rules in `docs/` when closed.

## Testing Strategy

- Existing `testKern` compound and tensor cases (regression for CIP-0003).
- Full matio Reference suite.
- Compile under C++17 (already the project floor).

## Related

- [CIP-0003](cip0003.md) tensor slash/copy fix
- Review note: manual kernel ownership (2026-11-13)
