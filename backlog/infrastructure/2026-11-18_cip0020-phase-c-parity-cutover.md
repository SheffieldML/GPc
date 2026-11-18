---
id: "2026-11-18_cip0020-phase-c-parity-cutover"
title: "CIP-0020 Phase C: matio parity and CI cutover decision"
status: "Completed"
priority: "Low"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "infrastructure"
related_cips: ["0020"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0020-phase-a-thin-cmake"]
tags:
- backlog
- infrastructure
- cmake
- ci
- cip0020
---

# Task: CIP-0020 Phase C — parity and cutover decision

## Description

Optionally extend CMake to the matio hard-gate test set and decide whether Actions should build with CMake (in parallel with make for one soak) or keep make forever for native CI.

## Acceptance Criteria

- [x] CMake/`ctest` (or scripted) coverage of the `run-matio-tests` set (`cmake/GpcTests.cmake`, 16 binaries; verified locally with `GPC_USE_MATIO=ON`)
- [x] Written cutover decision: **keep dual** — make remains Actions hard gate; CMake for packaging + optional maintainer ctest (no make retirement in CIP-0020)
- [x] No hard CI cutover while make and CMake disagree on flags or results (Actions unchanged)

## Implementation Notes

Assert-and-Verify: reference fixtures still gate merges via makefile workflows.

## Related

- CIP: 0020

## Progress Updates

### 2026-11-18

Task created Proposed; not started.

### 2026-11-18 (later)

Completed: `ctest` 16/16 passed with matio; cutover decision documented in CIP-0020 and `docs/build-and-ci.md`.
