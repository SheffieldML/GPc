---
id: "2026-11-18_cip0023-phase1-compile-skeleton"
title: "CIP-0023 Phase 1: ppa.h, base-class rebase, make ppa, CI smoke"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["0023"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- ppa
- build
- cli
---

# Task: CIP-0023 Phase 1 — compile skeleton

## Description

Restore `ppa.h`, rebase `CPpa` onto IVM-like bases so the sources link, add `make ppa`, and smoke `./ppa -h` in CI. Do not claim a working EM algorithm yet.

## Acceptance Criteria

- [x] `ppa.h` defines `CClppa` (mirror of `CClivm`)
- [x] `CPpa` compiles under C++17 against current `CKern` / `CNoise` / optimiser interfaces
- [x] `make ppa` produces `./ppa`
- [x] `./ppa -h` exits 0
- [x] Compile workflow smoke includes `./ppa -h`

## Related

- CIP: 0023
- REQ: 0002, 0004

## Progress Updates

### 2026-11-18
Task created; Phase 0 fixtures green. `make ppa` and `./ppa -h` succeed; CI compile.yml updated. EM port remains Phase 2.
