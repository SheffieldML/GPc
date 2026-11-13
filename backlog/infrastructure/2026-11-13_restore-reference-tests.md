---
id: "2026-11-13_restore-reference-tests"
title: "Restore runnable reference test suite"
status: "Completed"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "infrastructure"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: ["2026-11-13_macos-make-and-gp-build"]
tags:
- backlog
- testing
- matlab
- infrastructure
---

# Task: Restore reference tests

## Description

Make `make tests` (or a documented subset) build and run against `matfiles/`. MATLAB linkage is allowed for this task; retiring it is REQ-0003 / a later release.

## Acceptance Criteria

- [x] Documented steps build the reference test binaries
- [x] Kernel and matrix tests run with clear pass/fail
- [x] README notes how to run verification and the temporary MATLAB dependency

## Related

- CIP: 0001
- REQ: 0002

## Progress Updates

### 2026-11-13

Task created (Ready).

### 2026-11-13 (later)

Completed via CIP-0001 matio gate and CIP-0003 hard-gate cleanup.
