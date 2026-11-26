---
id: "2026-11-26_svgp-lazy-full-data"
title: "CSvgp: lazy full-data matrices so deep-GP minibatch steps do not scale with N"
status: "Completed"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "infrastructure"
related_cips: ["002A", "0035"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- svgp
- deepgp
---

# Task: Lazy full-data K_uf / A / diag matrices in CSvgp

## Description

`CSvgp::updateKernelMatrices` recomputed `K_uf`, `A = L^{-1} K_uf`, `diag K` and `diag Q` over all N training points on every call. Deep-GP layer cores call it every minibatch step but use only the explicit-input API, so step cost grew with N. On full MNIST this was reported as about 65% of step time (880 ms/step against about 150 ms/step once removed). That timing came from a separate session and has not been reproduced on this implementation.

## Acceptance Criteria

- [x] `setLazyFullData(bool)` defers the N-sized matrices until the full-data API needs them; default stays eager.
- [x] Updating kernel or inducing state invalidates the cache.
- [x] `CDeepGp` layers use lazy mode.
- [x] Lazy and eager agree to 1e-10 on ELBO, gradients, natural-gradient steps, marginals and predictions (points and weight-space inducing; full, half and small batches).
- [x] The explicit-input API leaves the full-data matrices unbuilt.
- [x] `testSvgp`, `testDeepGp`, `testSviBayesGplvm` pass.
- [ ] Step-time measured on the MNIST setup against eager (not yet done).
- [ ] `checkGradients` on the lazy path (the test shows lazy equals eager, not finite-difference agreement).

## Implementation Notes

Full-data members of `CSvgp` are now `mutable` caches filled by `ensureFullData()`, called at the points that read `A`/`diagK`/`diagQ` (`formAB` large-batch path, `fillMarginals` on a full ordered batch, `elboGrad`). The constructor still builds them once. Branch `deepgp-lazy-full-data`, commit 257f1db; test is `lazyEquivalenceCase` in `testSvgp.cpp`.

## Related

- CIP: 002A, 0035

## Progress Updates

### 2026-11-26
Implemented and tested; two follow-up checks open (timing, `checkGradients`).
