---
id: "2026-11-19_cip002C-phase-c-publish"
title: "CIP-002C Phase C: docs + TestPyPI / PyPI publish path"
status: "Completed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "infrastructure"
related_cips: ["002C"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002C-phase-b-repair"]
tags:
- backlog
- infrastructure
- python
- pypi
- cip002C
---

# Task: CIP-002C Phase C — publish path

## Description

Lock the PyPI distribution name (`gpc` vs `sheffieldml-gpc`), document user `pip install` vs contributor editable install, dry-run on TestPyPI or GitHub Release assets, then publish to PyPI after maintainer sign-off (Trusted Publisher preferred).

## Acceptance Criteria

- [x] PyPI project name checked and recorded in `python/pyproject.toml` + README
- [x] User docs: `pip install …` with no toolchain; contributor docs keep `pip install -e "python/[test]"`
- [x] TestPyPI and/or GitHub Release dry-run succeeds (workflow `artifacts-only` + Release asset attach; TestPyPI/PyPI OIDC jobs ready pending Trusted Publisher registration)
- [x] Clean-venv checklist: install → import → tiny `GPRegression` optimize/predict (`python/scripts/smoke_installed_gpc.py`)
- [x] PyPI upload only after explicit maintainer sign-off; OIDC Trusted Publisher if feasible (`environment: pypi` / `testpypi`)

## Implementation Notes

Import name stays `gpc` even if the distribution name differs. Do not bundle `pods` / matplotlib into the core wheel.

Locked **`gpc`** (PyPI/TestPyPI 404 on 2026-11-19). Workflow: `.github/workflows/publish-wheels.yml`. Maintainer still needs one-time Trusted Publisher + GitHub Environments before the first TestPyPI/PyPI upload.

## Related

- CIP: 002C
- Depends: 2026-11-19_cip002C-phase-b-repair

## Progress Updates

### 2026-11-19

Task created Ready on CIP-002C Accept.

### 2026-11-19 (Phase C)

Implemented on `cip002C-binary-wheels`: name lock, publish workflow, smoke script, docs.
