---
id: "2026-11-26_cip003d-phase5-python"
title: "CIP-003D phase 5: Python facade, warnings and docs"
status: "In Progress"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_cip003d-phase4-wiring"]
tags:
- backlog
- deepgp
- weightspace
- hermite
---

# Task: CIP-003D phase 5: Python facade, warnings and docs

## Description

`inducing='hermite'` on `DeepGP` and `SparseGPRegression` (kernel `'mlp'` required). The facade reads the layout report and calls `warnings.warn` for W1-W4 unconditionally, and `describe()` / `str(model)` show the delivered layout. Document the trade-off against Gaussian ridges in `docs/tutorials/deepgp.md`.

## Acceptance Criteria

- [x] E1-E4 reach Python as exceptions with the named fields
- [x] W1, W2, W4 reach Python as a `UserWarning` from a constructed case; none fires on a well-conditioned control
- [ ] W3 reaches Python as a `UserWarning` from a constructed case. Not met: the captured fraction of an MLP layout stays at 0.67-0.83, so the 0.5 placeholder is unreachable from Python. W3 is tested in C++ with an explicit threshold; calibration moves to phase 6
- [x] `inducing_fixed` left at default resolves to fixed for Hermite and `str(model)` says so; explicit `inducing_fixed=False` raises E4
- [x] Tutorial states the trade-off: no smoothed-erf reading, well-conditioned \(K_{uu}\), the capacity table, and the hidden-layer caveat

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.

Phase 5 implemented (pending review). `SparseGPRegression`/`DeepGP` accept `inducing='hermite'`; `inducing_fixed` is now `None`-defaulted; warnings come from C++ (`hermiteWarnings()`) and are emitted by `gpc.models` as `UserWarning`; `describe()` and `str(model)` show the layout. Tests: `python/tests/test_hermite.py` (18), `testPhase5Warnings` in `testInducingHermite`.

Found and fixed during this phase: `ndlutil::randn` keeps a Box-Muller spare across `init_genrand`, so the Hermite layout depended on how many normals had been drawn earlier in the process (same seed, different layout). Added `ndlutil::resetRandn()` and call it in `CInducingHermite::makeLayout`; regression in `testDeterminism`. Layouts built by earlier phase tests changed accordingly; all gates still pass.
