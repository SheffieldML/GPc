---
author: "Neil Lawrence"
created: "2026-11-19"
id: "002B"
last_updated: "2026-11-19"
status: "Implemented"
compressed: false
related_requirements: ["0001", "0007"]
related_cips: ["0006", "0020", "0021"]
tags:
- cip
- build
- cmake
- windows
- msvc
- ci
- infrastructure
title: "Windows MSVC CMake compile smoke on GitHub Actions"
---

# CIP-002B: Windows MSVC CMake compile smoke on GitHub Actions

> CIPs describe HOW to achieve requirements (WHAT). This CIP delivers the Windows portability driver anticipated by [CIP-0020](./cip0020.md) without reopening the makefile-as-blessed-native decision, and without folding into [CIP-0021](./cip0021.md) (Ubuntu/macOS CMake soak / cutover).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add a **Windows** GitHub Actions job that builds core CLIs via the thin CMake path using **MSVC (cl) + Intel Fortran (`ifx`) + OpenBLAS**, and runs help smoke. Keep Ubuntu/macOS **makefile** Compile / Reference / Coverage as the hard blessed gates. Windows starts as a **soft gate** (`continue-on-error: true`) until green for a soak period, then becomes a hard compile smoke gate — still CMake-only on Windows; no `make.win` revival.

**Which requirements?** Extends [REQ-0001](../requirements/req0001_contemporary-build.md) and preserves [REQ-0007](../requirements/req0007_competitive-default-build.md) OpenBLAS semantics under CMake on Windows.

## Motivation

README and [docs/build-and-ci.md](../docs/build-and-ci.md) defer Windows: no maintained make fragment, no CI. [CIP-0006](./cip0006.md) kept Windows out of the three-OS matrix. [CIP-0020](./cip0020.md) introduced thin CMake partly so Windows need not invent `make.win`.

MSVC is the Windows ABI users and wheel builders expect. SciPy already runs MSVC + `ifx` + OpenBLAS on Actions; we follow that pattern so `ndlfortran.f` stays the single Fortran source (no `ndlfortran.c` / f2c revival).

## Detailed Description

### Locked decisions

1. **CMake is the Windows path** — no new blessed `make.win`. Makefile remains blessed on Ubuntu/macOS only.
2. **First CI toolchain: MSVC + Intel Fortran (`ifx`) + OpenBLAS** — install Fortran via `fortran-lang/setup-fortran` (Intel), C++ via `ilammy/msvc-dev-cmd` or `vswhere` + Ninja/`cl`. OpenBLAS via vcpkg (or equivalent binary that exports Fortran-ABI BLAS/LAPACK symbols). Keep compiling `ndlfortran.f` (CIP-0020 hygiene: do not revive `ndlfortran.c` for this path).
3. **MinGW / MSYS2 is deferred** — useful fallback if MSVC+ifx proves brittle; not the first matrix entry.
4. **No CLAPACK / `blaswrap.h`** — historical `_MSC_VER` branch in `lapack.h` includes missing `blaswrap.h`. Gate that include behind an explicit `GPC_USE_CLAPACK` (off by default). OpenBLAS Fortran ABI matches the Unix path.
5. **Platform define** — set `GPC_PLAT` to `_WIN32` when `WIN32`. Stop falling through to `_LINUX`.
6. **Compiler flags** — gate GCC-only Fortran options (`-fPIC`, `-funroll-all-loops`, `-std=legacy`) so they apply only under GNU Fortran; use ifx-compatible flags (or none) under Intel.
7. **Workflow** — separate job on `windows-2022` (or `windows-latest`) in `compile.yml` (or dedicated workflow). Soft gate first (`continue-on-error: true`).
8. **Smoke only** — build `gp` (+ extra CLIs if cheap) + help smokes. No matio, no coverage.
9. **Rejected:** `make.win`; hard-gate day one; Windows matio CI; dual MinGW+MSVC before MSVC is green; cutting over Unix make CI.

### Effort sketch

| Phase | Scope | Effort |
| --- | --- | --- |
| A | CMake Windows/MSVC hygiene + soft Actions job + docs | ~1–3 days |
| B | Soft → hard after soak | hours + soak |
| C (later CIP) | MinGW fallback and/or matio on Windows | separate |

### Alternatives considered

- **MSYS2 MinGW first** — simpler Fortran, weaker MSVC ABI coverage. Deferred as Phase C / sibling.
- **`ndlfortran.c` + libf2c under MSVC** — conflicts with CIP-0020 Fortran hygiene. Rejected.
- **Fold into CIP-0021** — orthogonal (Unix cutover vs Windows add). Keep separate.

## Implementation Plan

1. **CMake / source hygiene** — `GPC_PLAT=_WIN32`; gate `-fPIC` / GNU Fortran flags; disable clapack include unless `GPC_USE_CLAPACK`; FindBLAS OpenBLAS on Windows (vcpkg toolchain file or `CMAKE_PREFIX_PATH`).
2. **Actions** — MSVC env + `setup-fortran` (intel) + OpenBLAS + Ninja; configure Release; build; help smoke; soft gate.
3. **Docs** — README + `docs/build-and-ci.md` Windows CMake/MSVC recipe.
4. **Phase B** — remove soft gate after soak.

## Backward Compatibility

- No change to Ubuntu/macOS make CI.
- `lapack.h` clapack include becomes opt-in; tree never shipped `blaswrap.h`, so this is a fix.
- Unix `_LINUX` / `_OSX` unchanged.

## Testing Strategy

- Actions: configure + build under MSVC+ifx; run help smokes.
- Watch OpenBLAS DLL on `PATH` at runtime.
- No matio required for closure.

## Related Requirements

- [REQ-0001](../requirements/req0001_contemporary-build.md)
- [REQ-0007](../requirements/req0007_competitive-default-build.md)

## Implementation Status

- [x] CMake: `_WIN32` plat; gate GNU Fortran flags; clapack opt-in
- [x] Soft Windows MSVC+ifx CMake job (compile + help smoke)
- [x] Document Windows MSVC + CMake recipe
- [x] Follow-up: MSVC link — `gp_fortran` STATIC + ifx `/names:lowercase /assume:underscore`; `ppa` gets `CSparsePpa` + `CMltools`
- [ ] Soft → hard after soak (Phase B)

## References

- [CIP-0020](./cip0020.md), [CIP-0021](./cip0021.md), [CIP-0006](./cip0006.md)
- SciPy Windows MSVC + ifx + OpenBLAS Actions pattern
- `fortran-lang/setup-fortran` (Intel)
- Historical `_MSC_VER` notes in `ndlutil.h` / `lapack.h`
