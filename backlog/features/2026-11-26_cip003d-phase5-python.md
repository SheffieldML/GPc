---
id: "2026-11-26_cip003d-phase5-python"
title: "CIP-003D phase 5: Python facade, warnings and docs"
status: "Ready"
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

- [ ] E1-E4 reach Python as exceptions with the named fields
- [ ] W1-W4 each reach Python as a `UserWarning` from a constructed case; none fires on a well-conditioned control
- [ ] `inducing_fixed` left at default resolves to fixed for Hermite and `str(model)` says so; explicit `inducing_fixed=False` raises E4
- [ ] Tutorial states the trade-off: no smoothed-erf reading, well-conditioned \(K_{uu}\), the capacity table, and the hidden-layer caveat

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.
