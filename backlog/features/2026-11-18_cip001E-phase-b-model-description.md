---
id: "2026-11-18_cip001E-phase-b-model-description"
title: "CIP-001E Phase B: ModelDescription + display formatters"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001E"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip001E-phase-a-optimiser-observer"]
tags:
- backlog
- display
- model
- api
---

# Task: CIP-001E Phase B model description

> Backlog task for [CIP-001E](../../cip/cip001E.md) Phase B.

## Description

Introduce `ModelDescription` / `ParamRow` builders for kern, noise, and `CGp`. Text formatter; migrate `display`; fix `CGp::display` writing only to the provided stream. Tests for paths/values and stream capture.

## Acceptance Criteria

- [x] `describe()` for `CKern` and `CGp` using GPy-canonical paths/units (`CNoise` still via existing `display` text in GP header)
- [x] Text formatter; `display(ostream&)` delegates to it
- [x] `CGp::display` does not leak to `cout`
- [x] Unit/smoke tests for description paths and redirected display
- [x] `.model` streams unchanged

## Related

- CIP: 001E
- Requirement: REQ-000F

## Progress Updates

### 2026-11-18

Task created (Ready); depends on Phase A.

### 2026-11-18 (done)

`CModelDescription.h`, `CKern`/`CGp` describe+display, tests in `testParamPaths` / `testCGp` green. Status → Completed.
