---
id: "2026-11-13_github-actions-compile-ci"
title: "Add GitHub Actions workflow that compiles core tools"
status: "In Progress"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "infrastructure"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: ["2026-11-13_rename-default-branch-main"]
tags:
- backlog
- ci
- build
- infrastructure
---

# Task: Add compile CI

## Description

Add a GitHub Actions workflow on Ubuntu that installs a C++/Fortran/LAPACK toolchain and builds `gp` (and preferably `ivm`/`gplvm`) so compilation regressions are caught on push/PR. MATLAB-dependent tests are out of scope for this task.

## Acceptance Criteria

- [ ] Workflow file under `.github/workflows/`
- [ ] Installs g++, gfortran, LAPACK/BLAS via apt
- [ ] Builds at least `gp` successfully on `ubuntu-latest`
- [ ] Runs on pushes and PRs to `main`
- [ ] Empty MATLAB include path does not break the compile (`-I` hazard fixed)

## Implementation Notes

Prefer `make.linux`-style `-llapack -lblas`. Fix `INCLUDE = $(MATLABINCLUDE)` when unset. Do not enable `_NDLMATLAB` in this workflow.

## Related

- CIP: 0001
- REQ: 0001

## Progress Updates

### 2026-11-13

Task created; implementing alongside branch rename.
