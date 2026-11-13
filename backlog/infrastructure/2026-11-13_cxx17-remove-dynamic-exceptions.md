---
id: "2026-11-13_cxx17-remove-dynamic-exceptions"
title: "Remove dynamic exception specs and build as C++17"
status: "Completed"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "infrastructure"
related_cips: ["0002"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cxx
- build
- infrastructure
---

# Task: C++17 language mode

## Description

Strip dynamic exception specifications from `CNdlInterfaces.h` and replace empty `throw()` on exception `what()`/destructors with `noexcept` in `ndlexceptions.h`. Switch `make.linux` to `-std=c++17` and keep Compile CI green.

## Acceptance Criteria

- [x] No `) throw(` dynamic specs remain in build headers
- [x] `ndlexceptions` uses `noexcept` for `what()` / relevant dtors
- [x] `-std=c++17` in make.linux / make.inc
- [x] GitHub Compile workflow green on this branch

## Related

- CIP: 0002
- REQ: 0004

## Progress Updates

### 2026-11-13

Task created; edits underway on `cip0002-cxx17-modernization`.

Compile CI green on PR https://github.com/SheffieldML/GPc/pull/14 — marked Completed.
