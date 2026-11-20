---
id: "2026-11-19_cip0026-phase-a-csparseppa"
title: "CIP-0026 Phase A: land CSparsePpa + testSparsePpa gate"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0026"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- ppa
- sparse
- dtc
---

# Task: CIP-0026 Phase A — CSparsePpa

> Backlog for [CIP-0026](../../cip/cip0026.md) Phase A.

## Description

Land the prototype `CSparsePpa` EM driver (CGp-backed DTC) and `testSparsePpa` as a matio-free verification gate. Document no-white / scalar-β convention from prototype notes.

## Acceptance Criteria

- [x] `CSparsePpa.{h,cpp}` build via makefile
- [x] `make testSparsePpa && ./testSparsePpa` green (DTC M=N vs dense, FTC remap, M&lt;N smoke, white-kernel drift doc)
- [x] Dense `CPpa` / `./ppa` / `testPpa` unchanged and still green
- [x] CIP-0026 marked In Progress; Phase A checklist item done

## Implementation Notes

- Full-data recovery is DTC with M=N, X_u=X (not CGp::FTC).
- Do not use CWhiteKern in M=N recovery tests (cross-covariance is zero).

## Related

- CIP: 0026
- Notes: [`cip/cip0026/prototype_notes.md`](../../cip/cip0026/prototype_notes.md)

## Progress Updates

### 2026-11-19

Task created and completed on branch `cip0026-sparse-ppa`.
