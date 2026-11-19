---
id: "2026-11-19_cip002B-cmake-msvc-hygiene"
title: "CIP-002B: CMake MSVC/Windows hygiene (plat, flags, clapack)"
status: "In Progress"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002B"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cmake
- windows
- msvc
---

# Task: CMake MSVC/Windows hygiene

## Description

Prepare the thin CMake tree for MSVC + Intel Fortran + OpenBLAS: correct platform define, gate GNU-only Fortran flags, and stop including missing `blaswrap.h` under `_MSC_VER` unless explicitly opted in.

## Acceptance Criteria

- [ ] `WIN32` builds define `GPC_PLAT=_WIN32` (not `_LINUX`)
- [ ] `-fPIC` / `-std=legacy` / `-funroll-all-loops` apply only under GNU Fortran
- [ ] `lapack.h` clapack/`blaswrap.h` path is behind `GPC_USE_CLAPACK` (default off)
- [ ] `FindBLAS`/`FindLAPACK` with OpenBLAS documented for Windows vcpkg/`CMAKE_PREFIX_PATH`

## Implementation Notes

Keep compiling `ndlfortran.f` (no `ndlfortran.c` revival).

## Related

- CIP: 002B

## Progress Updates

### 2026-11-19

Task created (Ready) on CIP-002B accept. Implementation started on `cip002B-windows-msvc` worktree: `GPC_PLAT=_WIN32`, GNU-only Fortran flags, `GPC_USE_CLAPACK` opt-in in `lapack.h`.
