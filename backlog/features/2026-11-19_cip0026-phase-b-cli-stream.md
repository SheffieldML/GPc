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

Optional sparse learn path on `./ppa`: `--approx dtc` (and related) with `-a M` inducing size; quiet CGp logging; bump `ppaVersion` / stream fields when persisting sparse models (or document that sparse models are not yet streamable).

## Acceptance Criteria

- [x] `./ppa learn` accepts approx + active-set size for sparse path without breaking dense default
- [x] Help text documents sparse flags and no-white / β convention
- [x] Sparse models refuse save with a clear error (stream format deferred)
- [x] Dense learn/test/predict unchanged

## Implementation Notes

- Link `ppa` target to `CGp.o` / inducing only if sparse learn is wired; keep dense path allocation-light where possible.
- Prefer inducing fixed by default for Phase B; free inducing is Phase D.

## Related

- CIP: 0026

## Progress Updates

### 2026-11-19

Task created and completed: `-A` / `-a` on `learn`; sparse omits white; refuse `.model` write.
