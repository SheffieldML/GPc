---
id: "2026-11-20_cip0030-phase4-surfaces"
title: "CIP-0030 Phase 4: CLI, stream, docs/smoke for SVI Bayesian GPLVM"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0030-phase3-probit-ordered"]
tags:
- backlog
- svgp
- svi
- bayesian-gplvm
- cli
- stream
---

# Task: CIP-0030 Phase 4 — surfaces (CLI / stream / docs)

## Description

Expose `CSviBayesGplvm` through `./bgplvm --inference svi`, add stream type `svibayesgplvm`, smoke coverage, and a tutorial. Python façade and MRD with shared \(q(\mathbf X)\) are follow-on backlog items.

## Acceptance Criteria

- [x] `CSviBayesGplvm` implements `CStreamInterface` (`svibayesgplvm`)
- [x] `bgplvm learn --inference svi` with batch / samples / inducing / Adam flags
- [x] `display` / `gnuplot` load both collapsed and SVI models
- [x] Stream round-trip in `testSviBayesGplvm`; smoke covers SVI path
- [x] Tutorial `docs/tutorials/bgplvm_svi.md`
- [x] Python façade and MRD deferred to follow-on tasks

## Related

- CIP: 0030
- Follow-on: `2026-11-20_cip0030-phase4b-python-facade`
- Follow-on: `2026-11-20_cip0030-phase4c-mrd-shared-qx`

## Progress Updates

### 2026-11-20

Phase 4 CLI/stream/docs landed. Python bind and MRD remain follow-ons.
