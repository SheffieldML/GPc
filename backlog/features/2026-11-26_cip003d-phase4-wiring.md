---
id: "2026-11-26_cip003d-phase4-wiring"
title: "CIP-003D phase 4: CGp, CDeepGp, stream and CLI wiring"
status: "Completed"
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

- [x] Stream round-trip: `K_{uu}` and predictions equal under `equals` (`ndlutil::MATCHTOL`)
- [x] E6: stream with inconsistent `V`, `ks` or `numActive`, or invalid `ks`, throws `StreamFormatError`
- [x] E3 and E4 raised from the C++ setters where they apply
- [x] Old `.model` files without a hermite block still load

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.

### 2026-11-26 (completed)

Wiring in `CGp`, `CSvgp`, `CDeepGp`, streams and both CLIs. The phase 3 deferral is closed: `CDeepGp` layer gradients agree with finite differences (62 params, about 1e-9). Tests in `testInducingHermite` cover E1-E6 at the C++ setters, `CGp` objective gradient (about 3e-9), `CGp` and `CDeepGp` stream round trips (V, ks, K_uu, K_ux and ELBO), and a points-model stream loading unchanged. CLI smoke: `deepgp learn -k mlp --inducing hermite` trains and reloads; `-k rbf` gives E2 and `-M 20` at d=1 gives E1 with the capacity table; `gp learn -A dtcvar --inducing hermite` trains, saves and reloads, and `--window-width` gives E3. The `gp` CLI sets inducing fixed for Hermite and prints a note saying so; the Python façade must instead refuse an explicit `inducing_fixed=False` (E4 at the façade).
