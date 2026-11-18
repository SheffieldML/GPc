---
id: "2026-11-18_tidy-repo-root"
title: "Tidy repo root: scratch demos, PNG plots, legacy make.* fragments"
status: "Proposed"
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

- [ ] Inventory root-level non-source artefacts (plots, diag/genTest, obsolete make fragments)
- [ ] Propose target directories without breaking `makefile` / CI defaults
- [ ] Move or delete with makefile / `.gitignore` updates as needed
- [ ] `make USE_MATIO=1 run-matio-tests` still green after the move

## Implementation Notes

Prefer a small CIP if the move touches many makefile rules; otherwise execute as this backlog task on a dedicated branch.

## Related

- Blessed make: `make.linux`, `make.osx` → `make.inc`
- CI: `.github/workflows/`

## Progress Updates

### 2026-11-18

Task created from external review (Claude). Proposed — pause before large moves.
