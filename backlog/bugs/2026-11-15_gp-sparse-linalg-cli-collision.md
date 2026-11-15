---
id: "2026-11-15_gp-sparse-linalg-cli-collision"
title: "gp -L collides Learn-scales with sparse-linalg"
status: "Proposed"
priority: "Medium"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "bugs"
related_cips: ["000E", "0012"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- bugs
- cli
- gp
- sparse
---

# Task: Fix `gp -L` flag collision

## Description

`gp.cpp` binds `-L` to both `--Learn-scales` and `--sparse-linalg`. The first match wins, so `gp -L stable` never selects StableFactor. Only `--sparse-linalg stable` works, while `docs/numerics.md` still documents the broken short form.

Tracked from [CIP-0012](../../cip/cip0012.md) F5 / CIP-000E CLI note.

## Acceptance Criteria

- [ ] Distinct short options (e.g. keep `-L` for Learn-scales; use another letter or long-only for sparse-linalg) — document the choice
- [ ] `gp … --sparse-linalg stable` still works; short form either works or is removed from help/docs
- [ ] `docs/numerics.md` and `gp` help strings agree

## Related

- CIP: 000E, 0012
- Code: `gp.cpp` argument parsing (~lines 135 and 151) and help
