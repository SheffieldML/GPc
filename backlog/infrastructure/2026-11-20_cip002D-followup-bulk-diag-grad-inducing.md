---
id: "2026-11-20_cip002D-followup-bulk-diag-grad-inducing"
title: "CIP-002D follow-up: bulk diag gradients, inducing ∂Z batching, MLP bulk path"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-22"
category: "infrastructure"
related_cips: ["002D", "0028"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- kernels
- allocations
- inducing
---

# Task: CIP-002D follow-up — bulk diag grads, inducing ∂Z, MLP bulk

## Description

Post-merge re-measurement of CIP-002D on `main` (2026-11-20) confirmed the Phase 1–4 speed-ups are real (default build ≈ release; sparse release ~2–13× vs pre-CIP release). Three QW gaps remain:

1. **Bulk diagonal parameter gradients (QW7 gap).** Allocations still scale ≈ \(2N\) per evaluation (`CMatrix::totalAlloc` ~10k at \(N=5000\), ~40k at \(N=20000\)). Cause: default `CKern::getDiagGradParams` loops over every data point, builds a \(1\times1\) problem, and calls the general gradient path; `CCmpndKern::getGradParams` allocates two temporaries per call. Add a bulk diagonal gradient API. For stationary kernels the diagonal is constant (lengthscale grad \(0\); variance grad is a sum). Also fix the latent trap: the per-point loop would multiply the kernel prior’s gradient by \(N\) if `regularise=true` (current callers pass `false`).

2. **Batched inducing-point \(\partial Z\) (QW5 gap).** `CInducing` still calls `getGradX` once per inducing point.

3. **Bulk path for MLP / directional features (QW5 deferred).** Unblocked now that CIP-0028 has merged; add the bulk cross-covariance / gradient path for those kernels.

## Acceptance Criteria

- [x] Bulk `getDiagGradParams` (or equivalent) removes the per-point \(1\times1\) loop; stationary kernels use the closed form
- [x] `regularise=true` applies the prior once, not \(N\) times (regression test)
- [x] Allocation gate: `CMatrix::totalAlloc` per sparse evaluation is \(O(1)\) in \(N\) (or a documented small constant), pinned by a test or `benchGPc` check
- [x] Inducing \(\partial Z\) uses a batched `getGradX` / GEMM-friendly path (no per-inducing `getGradX` in the hot loop)
- [x] MLP (and CIP-0028 directional/weight-space features as applicable) have a bulk path exercised by existing or new tests
- [ ] Likelihoods unchanged to printed / `MATCHTOL` digits on the CIP-002D timing cases; suites green under `CHECKS=full` and `RELEASE=1`

## Implementation Notes

- Priority order: (1) bulk diag + alloc pin, (2) inducing ∂Z, (3) MLP bulk.
- Re-measurement reference (ms/eval, same 2-core / `OPENBLAS_NUM_THREADS=1` setup as CIP-002D E1):

| Case | Pre default | Pre release | Post default | Post release |
| --- | --- | --- | --- | --- |
| FTC, \(N=1000\) | 15.7 | 3.2 | 2.9 | 3.3 |
| DTCVAR, \(N=5000, M=100\) | 2097 | 152 | 67 | 63 |
| FITC, \(N=5000, M=100\) | 1765 | 141 | 36 | 45 |
| DTCVAR, \(N=20000, M=200\) | — | 6761 | 522 | 526 |

## Related

- CIP: 002D (follow-up to Phases 2–3 QW5/QW7), 0028 (MLP / weight-space bulk unblocked)
- PRs: #89 (CIP-002D merge)

## Progress Updates

### 2026-11-20

Task created from post-merge re-measurement comments. CIP-002D E1/E2 re-measure recorded as done; these three items stay open.

### 2026-11-20 (item 1)

Bulk `getDiagGradParams` landed on branch `cip002D-bulk-diag-grad`:

- Stationary leaves: closed form (`K_ii = variance` → variance gets `sum(w)`, lengthscales 0)
- `CCmpndKern` / `CTensorKern` overrides (product rule for tensor); `CLinKern` closed form
- `regularise` applied once; pinned by `testDiagGradParams`
- Alloc pin: compound rbf+bias+white reports `totalAlloc=3` at both \(N=128\) and \(N=2048\)

### 2026-11-20 (items 2–3)

- `CKern::inducingGradZ` with RBF/Lin GEMM overrides; `CCmpndKern` sums components; wired from `CInducingPoints::gradFromKuuKuf`
- `CMlpKern::compute` bulk: `syrk`/`gemm` + asin map (element oracle retained)
- Pinned by `testInducingGradZ` (bulk vs per-inducing `getGradX`; MLP compute vs `computeElement`; inducing wire)

CIP-002D timing re-measure / `RELEASE=1` soak remains the last open acceptance tick.

### 2026-11-22

DeepGP profiling showed MLP `getGradParams` / `getGradX` still dominate step time (50–70% with MLP kernel). Bulk `compute` alone is not enough. Remaining work tracked as [2026-11-22_mlp-kern-bulk-grads](./2026-11-22_mlp-kern-bulk-grads.md).
