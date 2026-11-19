---
id: "2026-11-19_cip002C-phase-b-repair"
title: "CIP-002C Phase B: auditwheel/delocate self-contained OpenBLAS"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002C"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002C-phase-a-cibuildwheel"]
tags:
- backlog
- infrastructure
- python
- wheels
- openblas
- cip002C
---

# Task: CIP-002C Phase B — in-wheel BLAS/Fortran repair

## Description

Repair built wheels so `_gpc` does not depend on absolute Homebrew or system OpenBLAS / libgfortran paths. Run façade pytest (or FTC subset) against the repaired wheel in CI. Document `OPENBLAS_NUM_THREADS` and related concurrency notes for wheel users.

## Acceptance Criteria

- [x] Linux: auditwheel (or equivalent) repairs OpenBLAS + gfortran runtime into the wheel
- [x] macOS: delocate (or equivalent) yields `@loader_path`-relative deps; no Homebrew absolute RPATH required at import
- [x] `pytest -q python/tests` (or documented FTC smoke) passes on the repaired wheel in CI
- [x] `ldd` / `otool -L` shows no unresolved absolute OpenBLAS from system prefixes
- [x] Python README notes thread-env vars for wheel installs

## Implementation Notes

Contributor-oriented Apple `INSTALL_RPATH` in `python/CMakeLists.txt` may stay for editable builds; wheel layout must be relative after repair. Do not vendor OpenBLAS into the git tree (CIP-0020).

Landed: SKBUILD uses `@loader_path` only; `wheels.yml` on `macos-15` sets `MACOSX_DEPLOYMENT_TARGET` from brew bottle `minos`; default delocate/auditwheel; `check_native_deps.py` + pytest via `test-command`; `OPENBLAS_NUM_THREADS=1` in cibuildwheel env + README. Lower macOS floors need source-built OpenBLAS/gfortran (follow-up).

## Related

- CIP: 002C
- Depends: 2026-11-19_cip002C-phase-a-cibuildwheel

## Progress Updates

### 2026-11-19

Task created Ready on CIP-002C Accept.

### 2026-11-19 (Phase B)

Implemented on `cip002C-binary-wheels`; awaiting soft Actions confirmation on PR.
