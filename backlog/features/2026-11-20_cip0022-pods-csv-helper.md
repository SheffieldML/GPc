---
id: "2026-11-20_cip0022-pods-csv-helper"
title: "CIP-0022: cross-link Phase E / optional pods→CSV helper"
status: "Ready"
priority: "Low"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0022", "001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip001D-phase-e-notebooks"]
tags:
- backlog
- csv
- pods
- docs
- python
---

# Task: CIP-0022 pods→CSV helper / Phase E cross-link

## Description

CIP-0022 C++ CSV path, examples, SVML parity, and README/tenet soft-update are done. The deferred Implementation Status item is:

- Cross-link CIP-001D Phase E / optional pods→CSV helper (Phase E backlog owns notebooks)

## Acceptance Criteria

- [ ] Document pods as the interesting-data path from CIP-0022 / REQ-0011 into Phase E notebooks
- [ ] Optional tiny script dumps `pods` → CSV for offline CLI (under `scripts/` or `python/examples/`, not a core dep)
- [ ] CIP-0022 ready to Close once the cross-link is present

## Implementation Notes

Do not pull `pods` into core `gpc`. Prefer documenting the Phase E notebook path and a one-shot dump script if authors want offline CSV.

## Related

- CIP: 0022, 001D
- REQ: 0011
- Depends on: Phase E notebooks backlog

## Progress Updates

### 2026-11-20

Task created to clear VibeSafe In Progress → backlog coverage; CSV core delivery already Implemented.
