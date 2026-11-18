---
id: "2026-11-18_cip001E-phase-c-python-progress-repr"
title: "CIP-001E Phase C / CIP-001D B2: live progress + repr bindings"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001E", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip001E-phase-a-optimiser-observer", "2026-11-18_cip001E-phase-b-model-description"]
tags:
- backlog
- python
- pybind11
- optimiser
- display
---

# Task: CIP-001E Phase C Python live progress and repr

> Backlog task for [CIP-001E](../../cip/cip001E.md) Phase C / [CIP-001D](../../cip/cip001D.md) Phase B2.

## Description

Bind `OptimiserEvent` in pybind11; wire `optimize(messages=..., callback=...)` on `GPRegression`. Expose `__str__` / `__repr__` from `ModelDescription`. Pytest for callback cadence and silent opt-out.

## Acceptance Criteria

- [x] `OptimiserEvent` readable from Python
- [x] `optimize(..., messages=, callback=)` installs outer-iteration observer only
- [x] Opt-out (`messages=False`, no callback) → zero Python traffic during run
- [x] `str(m)` / `repr(m)` include canonical path labels
- [x] Pytest green

## Related

- CIP: 001E, 001D
- Requirement: REQ-000F, REQ-000E

## Progress Updates

### 2026-11-18

Task created (Ready); depends on Phases A–B.

### 2026-11-18 (done)

pybind bindings + `test_optimiser_display.py` green. Status → Completed.
