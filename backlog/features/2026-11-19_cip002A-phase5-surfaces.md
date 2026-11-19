---
id: "2026-11-19_cip002A-phase5-surfaces"
title: "CIP-002A Phase 5: CLI, stream, Python façade hook, docs/tutorial"
status: "In Progress"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["002A", "001D", "0014"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip002A-phase4-non-gaussian"]
tags:
- backlog
- svgp
- cli
- docs
---

# Task: CIP-002A Phase 5 — surfaces

## Description

Expose SVGP through CLI (`gp learn --approx svgp` or a dedicated binary), stream version for model I/O, a thin Python façade hook (CIP-001D), and a short tutorial. Parameter paths per CIP-0014.

## Acceptance Criteria

- [ ] CLI trains a small Gaussian SVGP end-to-end
- [ ] Stream round-trip preserves kernel, Z, β, and whitened \(q\)
- [ ] Tutorial documents natgrad + Adam usage and the DTCVAR oracle

## Related

- CIP: 002A
- REQ: 0015

## Progress Updates

### 2026-11-19
Task created on CIP acceptance.

### 2026-11-19
Tutorial `docs/tutorials/svgp.md` landed. Remaining: CLI flag, stream I/O, Python façade hook.
