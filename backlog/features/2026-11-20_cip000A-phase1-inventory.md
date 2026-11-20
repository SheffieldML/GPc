---
id: "2026-11-20_cip000A-phase1-inventory"
title: "CIP-000A Phase 1: Inventory owning vs non-owning CKern*"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["000A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- kern
- memory
- ownership
---

# Task: CIP-000A Phase 1 ownership inventory

> Backlog task for [CIP-000A](../../cip/cip000A.md) Phase 1.

## Description

Document which `CKern*` sites own versus observe. Primary owning site is `CComponentKern::components`. Factory paths (`readKernFromStream`, MATLAB `extractParamFromMxArray`) currently allocate then pass through cloning `addKern`, leaking the temporary.

## Acceptance Criteria

- [x] Owning vs non-owning table written into CIP-000A Detailed Description
- [x] Stream / MATLAB leak class noted for Phase 2/3 take-ownership overload
- [x] Out-of-scope sites (`CGp::pkern`, etc.) listed as deferred

## Implementation Notes

- Prefer documenting in the CIP itself (single source of truth).
- Soft-coordinate with CIP-0024 / CIP-0013 editors of `CKern.cpp`.

## Related

- CIP: 000A
- Requirement: REQ-0004

## Progress Updates

### 2026-11-20

Inventory completed and embedded in CIP-000A. Status → Completed.
