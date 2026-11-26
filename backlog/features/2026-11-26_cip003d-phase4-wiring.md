---
id: "2026-11-26_cip003d-phase4-wiring"
title: "CIP-003D phase 4: CGp, CDeepGp, stream and CLI wiring"
status: "Ready"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_cip003d-phase3-gradients"]
tags:
- backlog
- deepgp
- weightspace
- hermite
---

# Task: CIP-003D phase 4: CGp, CDeepGp, stream and CLI wiring

## Description

Add `"hermite"` to `CGp::setInducing`, `CDeepGp::setLayerInducingType`, a versioned stream block holding `V` and `ks` explicitly, and CLI `--inducing hermite`.

## Acceptance Criteria

- [ ] Stream round-trip: `K_{uu}` and predictions equal under `equals` (`ndlutil::MATCHTOL`)
- [ ] E6: stream with inconsistent `V`, `ks` or `numActive`, or invalid `ks`, throws `StreamFormatError`
- [ ] E3 and E4 raised from the C++ setters where they apply
- [ ] Old `.model` files without a hermite block still load

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.
