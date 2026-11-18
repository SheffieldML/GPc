---
id: "2026-11-19_cip0026-phase-b-cli-stream"
title: "CIP-0026 Phase B: sparse PPA CLI flags and stream version"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0026"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0026-phase-a-csparseppa"]
tags:
- backlog
- ppa
- cli
- stream
---

# Task: CIP-0026 Phase B — CLI / stream

> Backlog for [CIP-0026](../../cip/cip0026.md) Phase B.

## Description

Optional sparse learn path on `./ppa`: `--approx dtc` (and related) with `-a M` inducing size; quiet CGp logging; `ppaVersion=0.3` sparse stream.

## Acceptance Criteria

- [x] `./ppa learn` accepts approx + active-set size for sparse path without breaking dense default
- [x] Help text documents sparse flags and no-white / β convention
- [x] Sparse models stream with `ppaVersion=0.3` (`writeSparsePpaToFile` / `readSparsePpaFromFile`)
- [x] Dense learn/test/predict unchanged

## Related

- CIP: 0026

## Progress Updates

### 2026-11-19

CLI + stream round-trip landed and tested.
