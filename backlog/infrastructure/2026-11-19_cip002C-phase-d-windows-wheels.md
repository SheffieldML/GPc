---
id: "2026-11-19_cip002C-phase-d-windows-wheels"
title: "CIP-002C Phase D: Windows wheels after CIP-002B"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002C", "002B"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002C-phase-c-publish", "2026-11-19_cip002B-windows-msvc-actions"]
tags:
- backlog
- infrastructure
- python
- wheels
- windows
- cip002C
---

# Task: CIP-002C Phase D — Windows wheels

## Description

Extend cibuildwheel to Windows using the MSVC + ifx + OpenBLAS path from CIP-002B. Repair DLLs into the wheel / package data. Same import + pytest smoke as Unix.

## Acceptance Criteria

- [ ] CIP-002B Windows compile smoke is green (soft→hard optional per that CIP)
- [ ] cibuildwheel builds a Windows wheel for supported CPython versions
- [ ] OpenBLAS / runtime DLLs are loadable from the installed wheel without a separate system BLAS install
- [ ] `import gpc` + pytest/FTC smoke pass on Windows in CI
- [ ] Docs mention Windows `pip install` alongside Unix

## Implementation Notes

Blocked on CIP-002B. Do not revive `ndlfortran.c`. Keep makefile Unix CI unchanged.

## Related

- CIP: 002C, 002B
- Depends: Phase C publish path; CIP-002B Actions task

## Progress Updates

### 2026-11-19

Task created Proposed on CIP-002C Accept; blocked on CIP-002B Windows smoke.
