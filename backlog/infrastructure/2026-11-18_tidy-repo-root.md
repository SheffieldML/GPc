---
id: "2026-11-18_tidy-repo-root"
title: "Tidy repo root: scratch demos, PNG plots, legacy make.* fragments"
status: "Completed"
priority: "Low"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "infrastructure"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- infrastructure
- housekeeping
- onboarding
---

# Task: Tidy the repository root

## Description

Roughly a hundred files sit in the repo root: tutorial PNG plots, legacy `make.*` fragments (Fedora Core 6, Cygwin, …), and loose `diag*.cpp` / `genTest*.cpp` scratch programs. This makes the project look abandoned at first glance even though build/CI is healthy. A light reorganisation (e.g. `examples/plots/`, `make/legacy/`, `tools/` or `scratch/`) would help onboarding without changing the blessed `make.linux` / `make.osx` path.

## Acceptance Criteria

- [x] Inventory root-level non-source artefacts (plots, diag/genTest, obsolete make fragments)
- [x] Propose target directories without breaking `makefile` / CI defaults
- [x] Move or delete with makefile / `.gitignore` updates as needed
- [x] `make USE_MATIO=1 run-matio-tests` still green after the move

## Implementation Notes

Delivered on `fix-silent-test-exit-readme` without a separate CIP:

| From root | To |
| --- | --- |
| `*.png` tutorial plots | `docs/images/` |
| `make.{atlas,cygwin,fc6,macports,ubuntu}` | `make/legacy/` |
| `diag*.cpp`, `genTest*.cpp` | `tools/` (binaries still at root) |

Blessed `make.linux` / `make.osx` stay at the root. README image `src=` paths, `makefile` tool rules, `codecov.yml`, and Coverage workflow excludes updated accordingly.

## Related

- Blessed make: `make.linux`, `make.osx` → `make.inc`
- CI: `.github/workflows/`
- Branch: `fix-silent-test-exit-readme`

## Progress Updates

### 2026-11-18

Task created from external review (Claude). Implemented the light reorganisation on the silent-test / README branch; marked Completed.
