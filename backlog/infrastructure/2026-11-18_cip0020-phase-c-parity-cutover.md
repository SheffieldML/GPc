---
id: "2026-11-18_cip0020-phase-c-parity-cutover"
title: "CIP-0020 Phase C: matio parity and CI cutover decision"
status: "Proposed"
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

- [ ] CMake/`ctest` (or scripted) coverage of the `run-matio-tests` set
- [ ] Written cutover decision: retire make, wrap make→CMake, or keep dual forever with clear owners
- [ ] No hard CI cutover while make and CMake disagree on flags or results

## Implementation Notes

Assert-and-Verify: reference fixtures still gate merges.

## Related

- CIP: 0020

## Progress Updates

### 2026-11-18

Task created Proposed; not started.
