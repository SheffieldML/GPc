---
id: "2026-11-18_cip0020-phase-a-thin-cmake"
title: "CIP-0020 Phase A: thin CMake for libgp + gp smoke"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "infrastructure"
related_cips: ["0020"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0020-phase-0-archive"]
tags:
- backlog
- infrastructure
- cmake
- build
- cip0020
---

# Task: CIP-0020 Phase A — thin CMake

## Description

Add a living thin CMake build that produces `libgp` and CLI smoke binaries (`gp`, optionally `ivm` / `gplvm` / `bgplvm`) with C++17, Fortran `ndlfortran.f`, OpenBLAS, and optional matio — without replacing makefile CI.

## Acceptance Criteria

- [x] `cmake -B build && cmake --build build` builds `gp`
- [x] `./build/gp -h` (or build-dir equivalent) smokes
- [x] OpenBLAS / LAPACK via CMake find packages (CIP-0013 policy)
- [x] `GPC_USE_MATIO` option maps to `_NDLMATIO`
- [x] Release build maps to `GPC_CHECKS=0` / `NDEBUG` like `make RELEASE=1`
- [x] Shared sources listed in `cmake/GpcSources.cmake`
- [x] Makefile path unchanged and still documented as blessed

## Implementation Notes

On macOS, prefer Homebrew `g++-N` / `gfortran-N` for ABI match (same rationale as `make.osx`).

## Related

- CIP: 0020

## Progress Updates

### 2026-11-18

Implemented and smoke-tested on the CIP worktree.
