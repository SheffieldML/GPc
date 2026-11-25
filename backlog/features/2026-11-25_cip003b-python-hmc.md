---
id: "2026-11-25_cip003b-python-hmc"
title: "CIP-003B: gpc.inference.hmc Python wrapper"
status: "Completed"
priority: "Medium"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["003B"]
owner: "Neil Lawrence"
dependencies: ["2026-11-25_cip003b-gp-sample-cli"]
tags:
- backlog
- hmc
- python
---

# Task: `gpc.inference.hmc` Python wrapper

## Description

Expose prior placement and HMC sampling of hyperparameters through the `gpc` façade for `GPRegression` and `SparseGPRegression`. Sampling stays in C++; NumPy arrays cross the boundary once, after the chain.

## Acceptance Criteria

- [x] `model.set_prior(path, spec)` and `model.hmc(...)` bindings; `gpc.inference.set_prior`, `gpc.inference.hmc` returning an `HmcResult` (names, samples in natural units, log posterior, accept rate, step size) with `to_csv`
- [x] Improper targets refused with `GpcError`; `allow_improper`, `include_jacobian`, `fix`, `seed` supported; model state restored afterwards
- [x] Python tests (`python/tests/test_hmc.py`) include a comparison against grid integration of the posterior
- [x] `CHmc` is in libgp (makefile and CMake) so the extension links

## Related

- CIP: 003B

## Progress Updates

### 2026-11-25
Implemented. 11 tests in `test_hmc.py` pass, and the full Python suite passes (113 passed, 3 skipped). Priors on `rbf.lengthscale` are refused because priors attach to the native inverse width (see the follow-up task).
