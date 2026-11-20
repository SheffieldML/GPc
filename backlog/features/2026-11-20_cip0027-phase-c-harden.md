---
id: "2026-11-20_cip0027-phase-c-harden"
title: "CIP-0027 Phase C: optional harden + soak (White + likelihood variance)"
status: "Ready"
priority: "Low"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0027"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0027-phase-bp-ivm-softener"]
tags:
- backlog
- noise
- deprecation
- soak
---

# Task: CIP-0027 Phase C — Harden dual-noise path

## Description

Optional `--allow-white-and-noise` for Gaussian regression when both White and likelihood variance are present. Soak; then drop dual interpretation if evidence supports it.

## Acceptance Criteria

- [ ] Flag (or equivalent) documented for intentional White + likelihood variance
- [ ] Soak notes recorded; dual-path behaviour tightened or deferred with rationale

## Related

- CIP: 0027
- Depends on: Phase B′

## Progress Updates

### 2026-11-20

Task created on Accept.
