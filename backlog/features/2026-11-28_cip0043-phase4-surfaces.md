---
id: "2026-11-28_cip0043-phase4-surfaces"
title: "CIP-0043 Phase 4: SVI GP-LVM, CLIs, gpc.kern, docs"
status: "Proposed"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "features"
related_cips: ["0043", "0030", "0038"]
owner: "Neil Lawrence"
dependencies: ["2026-11-28_cip0043-phase3-deepgp"]
tags:
- backlog
- relu
- cli
- python
- docs
---

# Task: CIP-0043 Phase 4 — remaining surfaces

## Description

Add ReLU to the SVI Bayesian GP-LVM kernel switch (`bgplvm.cpp`), with a typed refusal on the collapsed psi path. Add `-k relu` / `-k reluard` to `gp`, `gplvm`, `ivm` and `ppa` and list ReLU alongside MLP in the `-w` / `-b` help. Add `gpc.kern.ReLU` (and ARD), `kernel='relu'` aliases and wrappers in `wrap_gp.cpp`, `wrap_gplvm.cpp`, `wrap_deepgp.cpp`. Update docs.

## Acceptance Criteria

- [ ] `testSviBayesGplvm` has a ReLU case; the collapsed path raises a typed error.
- [ ] CLI help and parsing accept `relu` / `reluard`; a smoke run per CLI where one exists.
- [ ] Python tests beside `python/tests/test_mlp_kernels.py` cover `ReLU`; parameter paths work.
- [ ] `docs/deepgp.md`, `docs/numerics.md`, `README.md` and `python/README.md` list the kernel.

## Related

- CIP: 0043, 0030, 0038

## Progress Updates

### 2026-11-28

Task created, Proposed (blocked on Phase 3).
