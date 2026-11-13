---
id: "2026-11-13_macos-make-and-gp-build"
title: "Refresh macOS make config and build gp locally"
status: "Ready"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "infrastructure"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: ["2026-11-13_github-actions-compile-ci"]
tags:
- backlog
- build
- macos
- infrastructure
---

# Task: macOS make config and local gp build

## Description

Establish a working macOS `make.inc` (OpenBLAS/Accelerate + gfortran as available) and get `gp` linking and smoke-running locally for REQ-0001 beyond Linux CI.

## Acceptance Criteria

- [ ] Documented macOS make fragment works on this machine
- [ ] `makefile` includes via `make.inc` (or equivalent clear mechanism)
- [ ] `./gp -h` (or equivalent) runs after build

## Related

- CIP: 0001
- REQ: 0001

## Progress Updates

### 2026-11-13

Task created (Ready); starts after compile CI lands.
