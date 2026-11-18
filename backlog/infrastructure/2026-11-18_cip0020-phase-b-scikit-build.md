---
id: "2026-11-18_cip0020-phase-b-scikit-build"
title: "CIP-0020 Phase B: scikit-build wheels over thin CMake"
status: "Completed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "infrastructure"
related_cips: ["0020", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0020-phase-a-thin-cmake"]
tags:
- backlog
- infrastructure
- cmake
- python
- packaging
- cip0020
---

# Task: CIP-0020 Phase B — scikit-build / wheels

## Description

Wire CIP-001D’s Python façade packaging (`python/pyproject.toml`) to the thin CMake from Phase A via scikit-build-core (or equivalent), while keeping `python/Makefile` editable builds working until wheels are documented.

## Acceptance Criteria

- [x] `pyproject.toml` (or sibling) builds `_gpc` against CMake `libgp`
- [x] Documented wheel / editable install path for ordinary users
- [x] CIP-001D FTC smoke / pytest still pass via the new path
- [x] `python/Makefile` remains usable for day-to-day façade work until cutover note

## Implementation Notes

Do not cut over GitHub Actions Compile / Reference gates in this task (that is Phase C).

scikit-build-core with `cmake.source-dir = ".."`; `SKBUILD` disables CLI targets. Homebrew: set `CMAKE_PREFIX_PATH`, `CXX`, `FC` as in `python/README.md`.

## Related

- CIP: 0020, 001D

## Progress Updates

### 2026-11-18

Task created Ready after CIP-0020 Accept; blocked on packaging demand from CIP-001D.

### 2026-11-18 (later)

Completed: editable `pip install -e python/[test]` builds `_gpc`; `pytest python/tests` → 7 passed. Makefile path left intact.
