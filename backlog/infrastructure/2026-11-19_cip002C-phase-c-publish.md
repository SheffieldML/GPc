---
id: "2026-11-19_cip002C-phase-c-publish"
title: "CIP-002C Phase C: docs + TestPyPI / PyPI publish path"
status: "Ready"
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

- [ ] PyPI project name checked and recorded in `python/pyproject.toml` + README
- [ ] User docs: `pip install …` with no toolchain; contributor docs keep `pip install -e "python/[test]"`
- [ ] TestPyPI and/or GitHub Release dry-run succeeds
- [ ] Clean-venv checklist: install → import → tiny `GPRegression` optimize/predict
- [ ] PyPI upload only after explicit maintainer sign-off; OIDC Trusted Publisher if feasible

## Implementation Notes

Import name stays `gpc` even if the distribution name differs. Do not bundle `pods` / matplotlib into the core wheel.

## Related

- CIP: 002C
- Depends: 2026-11-19_cip002C-phase-b-repair

## Progress Updates

### 2026-11-19

Task created Ready on CIP-002C Accept.
