---
id: "2026-11-18_cip0020-phase-0-archive"
title: "CIP-0020 Phase 0: archive dead CMakeLists and document blessed make"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "infrastructure"
related_cips: ["0020"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- infrastructure
- cmake
- build
- cip0020
---

# Task: CIP-0020 Phase 0 — archive dead CMakeLists

## Description

Move the obsolete root `CMakeLists.txt` (2020 mlprojects/f2c layout) out of the way and document that makefile remains the blessed native/CI path while thin CMake is CIP-0020.

## Acceptance Criteria

- [x] Dead stub archived under `make/legacy/CMakeLists.txt.archive-2020` (not a supported build)
- [x] README / `docs/build-and-ci.md` state make is blessed; CMake is CIP-0020 optional path
- [x] New thin CMake (Phase A) uses `ndlfortran.f`, not `ndlfortran.c`

## Implementation Notes

Coordinate with Fortran hygiene backlog: do not revive `ndlfortran.c` for CMake.

## Related

- CIP: 0020

## Progress Updates

### 2026-11-18

Completed with Phase A thin CMake landing on branch `cip0020-cmake-when-packaging`.
