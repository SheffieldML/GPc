---
id: "2026-11-19_cip002B-windows-msvc-actions"
title: "CIP-002B: Soft Windows MSVC+ifx CMake compile job"
status: "In Progress"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002B"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002B-cmake-msvc-hygiene"]
tags:
- backlog
- ci
- windows
- msvc
- cmake
---

# Task: Soft Windows MSVC+ifx CMake Actions job

## Description

Add a soft-gated GitHub Actions job on `windows-2022` that configures thin CMake with MSVC (`cl`) + Intel Fortran (`ifx`) + OpenBLAS, builds core CLIs, and runs help smoke.

## Acceptance Criteria

- [ ] Job uses MSVC + `fortran-lang/setup-fortran` (intel) + OpenBLAS + Ninja (or VS generator)
- [ ] Builds at least `gp`; preferably extra CLIs
- [ ] Help smoke runs (`gp -h`, …) with OpenBLAS on `PATH` if needed
- [ ] `continue-on-error: true` (soft gate) until Phase B
- [ ] Ubuntu/macOS makefile Compile jobs unchanged
- [ ] README / `docs/build-and-ci.md` document the Windows recipe

## Implementation Notes

Follow SciPy-style MSVC + ifx + OpenBLAS. Soft gate first per CIP-002B.

## Related

- CIP: 002B
- Depends: 2026-11-19_cip002B-cmake-msvc-hygiene

## Progress Updates

### 2026-11-19

Task created (Ready) on CIP-002B accept. Soft `windows-msvc-cmake` job added to `compile.yml` (MSVC + ifx + vcpkg OpenBLAS); docs/README updated.
