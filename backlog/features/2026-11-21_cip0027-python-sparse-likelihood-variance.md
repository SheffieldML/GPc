---
id: "2026-11-21_cip0027-python-sparse-likelihood-variance"
title: "CIP-0027 follow-on: teach sparse Python noise as likelihood.variance"
status: "Ready"
priority: "Medium"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0027", "001D"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- python
- noise
- sparse
- likelihood
- gpy
---

# Task: Sparse façade — `likelihood.variance` first, not β

> Gap after closed [CIP-0027](../../cip/cip0027.md): C++ sparse `CGp` already syncs \(\beta \leftrightarrow\) `likelihood.variance` and lists both paths; the Python layer still *documents* and comments “FTC: likelihood.variance / sparse: β”.

## Description

Align the `gpc` sparse surface with the architecture / CIP-0027 teaching:

1. Prefer **`likelihood.variance`** (alias `Gaussian_noise.variance`) in README, examples, docstrings, and `wrap_gp.cpp` comments.
2. Keep **`beta`** as a sparse-only advanced/alias path if C++ still exposes it; do not lead with it for GPy users.
3. Pytest: sparse DTCVAR `set_by_path("likelihood.variance", …)` / `get_by_path` round-trip; objective matches setting via `beta` where both exist (\(σ^2 = 1/\beta\)).
4. Update `python/docs/subset_checklist.md` / side-by-side if they still imply sparse-only β.

No new sparse algebra — docs + façade teaching (+ tests). Internal \(\beta\) storage stays.

## Acceptance Criteria

- [ ] Sparse Python docs and binding comments lead with `likelihood.variance` / `Gaussian_noise.variance`
- [ ] Pytest: sparse get/set via `likelihood.variance`; equivalence with `beta` documented or asserted
- [ ] Weight-space / sparse README lines no longer say observation noise “stays in β” as the user-facing story
- [ ] No change to FTC behaviour (already CIP-0027-aligned)

## Implementation Notes

- C++ already: `paramPaths` includes `likelihood.variance` for Gaussian noise and `beta` when sparse; `getByPath`/`setByPath` sync ([`CGp.cpp`](../../CGp.cpp)).
- Touch: `python/src/wrap_gp.cpp`, `python/README.md`, `python/docs/*`, sparse tests under `python/tests/`.

## Related

- CIP: 0027, 001D
- REQ: 0014 (Validated; this is façade/doc follow-through)

## Progress Updates

### 2026-11-21

Task created Ready after Python-compat gap review (0027 follow-through, not a new CIP).
