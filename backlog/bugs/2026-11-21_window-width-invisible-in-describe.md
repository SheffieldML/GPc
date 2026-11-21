---
id: "2026-11-21_window-width-invisible-in-describe"
title: "inducing.window_width missing from param_paths / describe"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "bugs"
related_cips: ["0028", "001E"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_weightspace-learn-window-width"]
tags:
- backlog
- weightspace
- display
- inducing
---

# Task: Surface `inducing.window_width` in paths and model print

## Description

`learn_window_width=True` did pack and optimise global \(s\), but `param_paths()`, `str(m)`, and `describe()` omitted `inducing.window_width`. The printed "Number of Optimization Parameters" also under-counted free inducing centres, so learning \(s\) looked inert from the façade.

## Acceptance Criteria

- [x] `inducing.window_width` in `param_paths` / `get_by_path` / `set_by_path` for non-point inducing
- [x] `describe()` / `str(m)` show the row (`fixed` when not learnt, `+ve` when learnt)
- [x] Displayed optimisation parameter count uses `getOptNumParams()` (includes free centres + \(s\))
- [x] Tests cover fixed vs learnt visibility

## Progress Updates

### 2026-11-21

Fixed in `CGp::paramPaths` / `getByPath` / `setByPath` / `describe`, `ModelDescription.optNumParams`, and Python `_display` / pybind. Pytest green.
