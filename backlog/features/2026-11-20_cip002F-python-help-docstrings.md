---
category: features
completed: '2026-11-20'
created: '2026-11-20'
dependencies: []
id: 2026-11-20_cip002F-python-help-docstrings
last_updated: '2026-11-20'
owner: Neil Lawrence
priority: High
related_cips:
- 002F
status: Completed
title: 'CIP-002F: Python-compatible help/docstrings for gpc'
---

# Task: CIP-002F Python help and docstrings

> Backlog task for [CIP-002F](../../cip/cip002F.md) / [REQ-0017](../../requirements/req0017_python-facade-help-docstrings.md).

## Description

Add non-empty `__doc__` on the public first-subset surface so `help()`, notebook `?`, and IDE hover work. Cover `GPRegression`, `SparseGPRegression`, kern factories, and path/`optimize`/`predict` methods. Markdown side-by-side stays the porting narrative.

## Acceptance Criteria

- [x] Public constructors/methods have non-empty `__doc__` visible via `help()`
- [x] Pytest smoke asserts `__doc__` for the public subset
- [x] `python/README.md` points at `help()` as the interactive path
- [x] No modelling behaviour change

## Related

- CIP: 002F

## Progress Updates

### 2026-11-20

CIP-002F Accepted → Implemented → Closed on branch `cip002F-python-help-docstrings` (worktree path `GPc-cip001D`).