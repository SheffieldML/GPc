---
id: "2026-11-19_cip002C-phase-a-cibuildwheel"
title: "CIP-002C Phase A: cibuildwheel Linux + macOS soft gate"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002C"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- infrastructure
- python
- wheels
- cibuildwheel
- cip002C
---

# Task: CIP-002C Phase A — cibuildwheel soft gate

## Description

Add a soft-gated GitHub Actions workflow that builds Linux (manylinux) and macOS wheels for the `gpc` package via cibuildwheel + the existing scikit-build / thin CMake path, uploads artifacts, and smokes `import gpc` in a clean venv.

## Acceptance Criteria

- [x] `.github/workflows/wheels.yml` (or equivalent) runs cibuildwheel against `python/`
- [x] Matrix covers manylinux x86_64 and at least one macOS arch available on runners
- [x] Build image/job provides CMake, gfortran, and OpenBLAS for FindBLAS
- [x] Soft gate (`continue-on-error: true`) until green
- [x] Wheel artifacts uploaded
- [x] Clean-venv smoke: `python -c "import gpc"`
- [x] Native makefile Compile / Reference / Coverage jobs unchanged

## Implementation Notes

Consume CIP-0020 scikit-build (`python/pyproject.toml`, `GPC_BUILD_PYTHON`). Do not invent a second extension build. Phase B owns auditwheel/delocate self-containment.

Landed: `wheels.yml` + `[tool.cibuildwheel]` in `python/pyproject.toml` (CPython 3.10–3.13, `auto64`, skip win/musl/PyPy). Linux `before-all` installs `openblas-devel` + `gcc-gfortran`. macOS job installs Homebrew OpenBLAS/gcc and sets `CMAKE_PREFIX_PATH` / `CXX` / `FC`. Soft gate until first green Actions run.

## Related

- CIP: 002C
- Worktree: `GPc-cip002C` / `cip002C-binary-wheels`

## Progress Updates

### 2026-11-19

Task created Ready on CIP-002C Accept.

### 2026-11-19 (later)

Completed on branch `cip002C-binary-wheels`: workflow + pyproject cibuildwheel config + docs. Awaiting first soft Actions run after PR/push.
