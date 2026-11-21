---
id: "2026-11-21_weightspace-per-unit-window-width"
title: "Per-unit window widths s_m for weight-space ridges (CIP-0032)"
status: "Proposed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0032", "0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_weightspace-learn-window-width"]
tags:
- backlog
- weightspace
- mlp
- inducing
- window-width
---

# Task: Per-unit window widths \(s_m\) (CIP-0032)

## Description

CIP-0028 ridges use a shared global \(s\). That yields an MLP-shaped mean but not MLP-equivalent per-unit sharpness (a normal net grows \(\|\mathbf w_m\|\) per unit). [CIP-0032](../../cip/cip0032.md) proposes untied positive \(s_m\) with tied scalar as default.

## Acceptance Criteria

- [ ] CIP-0032 Accepted after review
- [ ] Phase A formulae + oracles landed
- [ ] Phase B C++ inducing / CGp / stream
- [ ] Phase C Python façade + docs + notebook

## Related

- CIP: 0032, 0028
- Sibling: `2026-11-21_weightspace-learn-window-width` (global \(s\) only)

## Progress Updates

### 2026-11-21

Task created with CIP-0032 draft (Proposed).
