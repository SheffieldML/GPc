---
id: "2026-11-15_stable-sparse-linalg"
title: "Stable sparse-GP factorisation dual-path (CIP-000E)"
status: "Completed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["000E"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- numerics
- dtcvar
- infrastructure
---

# Task: Stable sparse factorisation dual-path

## Description

Ship CIP-000E: Legacy vs StableFactor (`trsm`) path for \(K_{uu}^{-1}K_{uf}\), dual-path gate in `diagDtcvar`, CLI `-L`.

## Acceptance Criteria

- [x] Flag + `_multiplyInvKuu` for DTCVAR/FITC
- [x] `diagDtcvar` dual-path PASS within `MATCHTOL`
- [x] Docs in `docs/numerics.md`
- [x] CIP-000E Implemented after local green
- [ ] CIP-000E Closed after PR CI

## Progress Updates

### 2026-11-15

Implemented on `cip000E-sparse-linalg`. Local `diagDtcvar` and `testGp` green.
