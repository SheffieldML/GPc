---
author: "Neil Lawrence"
created: "2026-11-26"
id: "003C"
last_updated: "2026-11-26"
status: "In Progress"
compressed: false
related_requirements: ["0002"]
related_cips: ["0028", "0037", "003A"]
tags:
- cip
- deepgp
- python
- weightspace
- bug
title: "Fixes and gaps found reproducing the deep GP paper experiments through the Python façade"
---

# CIP-003C: Fixes and gaps found reproducing the deep GP paper experiments

> CIPs describe HOW to achieve requirements (WHAT). REQ-0002 covers runnable verification.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [ ] Implemented - Code changes complete
- [ ] Closed - Verified and merged
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Implementing the experiments of *Practical Deep Gaussian Processes* as eleven notebooks over `gpc` 0.1.0a11 (`deepgp-mlp/notebooks/`) exposed one defect and several façade gaps. This CIP lists them, with a reproduction and a proposed fix for each, so the notebooks can drop their workarounds and run the paper's full protocol.

## Motivation

Verification against the paper is the purpose of the notebooks (project tenet: assert invariants and verify against reference). Each item below either forced a workaround in `notebooks/common.py` or removed a comparison the paper makes.

## Detailed Description

### 1. Defect: weight-space construction fails for some (M, d)

`SparseGPRegression(..., kernel="mlp", inducing="weightspace")` and `DeepGP(..., inducing="weightspace")` raise `CInducingWeightSpace ridge direction has zero norm.` for some pairs of inducing count `M` and input dimension `d`. The failure does not depend on `N`, on the values of `Z`, or on the approximation. Reproduction (`X` standard normal, `Z = X[:M]`):

| d | fails at M | succeeds at M |
|---|---|---|
| 2 | 31, 32, 63, 64, 65, 127, 129, 255, 257 | 8, 16, 20, 24, 33, 40, 48, 96 |
| 8 | 63, 65, 127, 129, 255, 257 | 31, 33, 96 |

The error is thrown in `CInducingWeightSpace::ridgeFactors` (`CInducing.cpp`) when a row of `W` has zero weight-direction norm. Unconfirmed hypothesis: in `CGp.cpp` (around line 4030) `W` is zeroed and filled from `X_u` only when `X_u.getRows() == numActive && X_u.getCols() == d`; some path leaves rows at zero or uninitialised. Because the outcome varies erratically with `M`, uninitialised memory is also possible. Proposed fix: find the cause with a debug build and valgrind or ASan, fix it, and add a regression test that sweeps `M in 1..300` for `d in {1, 2, 8}`. The workaround in `common.py` (retry with `M+1..M+8`) is to be removed afterwards.

### 2. Gaps in the `DeepGP` façade

| Gap | Effect on the notebooks | Proposed change |
|---|---|---|
| Student-t likelihood unavailable (CIP-003A provides it in `CNoise`) | Label-outlier study in `09_robustness` runs with Gaussian likelihood only | Accept `likelihood='studentt'` where the Gaussian-expectation ELBO term has an implementation, or document why not |
| No Hermite features and no learned window widths | The paper's Hermite and learned-window variants are not run | Expose `learn_window_width(s)` and the Hermite option on `DeepGP`, as `SparseGPRegression` already exposes the former |
| Hidden width greater than input dimension raises `PCA width cannot exceed input dim` | `fit_deepgp` clips `hidden` to `d_in`, so 2-D problems cannot use a hidden width of 5 | Pad the PCA mean path (zero or random orthonormal directions), or state the constraint in the docstring and `deepgp.md` |
| `jacobian_spread` requires square input and output dimensions | `04_depth_defaults` uses D=8 in and out | Accept rectangular Jacobians (use singular values) |
| `optimize` chunked across calls restarts Adam state | Time-to-target curves in `10_cost` restart the optimiser at each checkpoint | Keep optimiser state between calls, or add a callback with checkpoints |
| Gaussian `predict` variance includes noise; `predict_noiseless` is not on `DeepGP` | Notebooks subtract nothing and report predictive NLL only | Add `predict_noiseless` to `DeepGP`, and state the convention in the docstring |

### 3. Data availability (outside GPc)

Power, protein and year are not on OpenML under stable names. The benchmark notebook omits them. Resolve through `pods.datasets` or a local fetch script; no GPc change.

## Implementation Plan

1. Reproduce item 1 in a C++ test, find the cause, fix, and add the regression sweep.
2. Remove the retry workaround from `notebooks/common.py` and re-run notebooks 03 and 05.
3. Implement the façade items in the order: `predict_noiseless`, rectangular `jacobian_spread`, PCA padding, Adam state, `learn_window_width(s)` and Hermite, Student-t.
4. Re-run the notebooks with `DGP_FAST=0` and compare with the paper's reported numbers.

## Backward Compatibility

Item 1 is a bug fix. The façade additions are new options with unchanged defaults. PCA padding changes behaviour only where construction currently raises.

## Testing Strategy

C++ regression sweep for item 1; Python tests for each façade item; `checkGradients` on any new likelihood path; the notebooks themselves as the end-to-end check.

## Related Requirements

- REQ-0002 (runnable verification)

## Implementation Status

- [x] Item 1: weight-space defect located and fixed (`makeRidgeInit` used `nDir = M / nOff`, rounding down, so trailing rows stayed zero; now `ceil`)
- [x] Item 1: regression sweep added (`testCGp`, M=1..300, d in {1,2,8})
- [x] `predict_noiseless` on `DeepGP`
- [x] Rectangular `jacobian_spread`
- [x] Hidden width above input dimension (PCA zero-padded)
- [x] Adam state kept across `optimize` calls (`CAdam` retains moments; `reset_optimizer()`)
- [x] Learned windows on `DeepGP` (`learn_window_width(s)`)
- [ ] Hermite features on `DeepGP` (not in GPc: the only Hermite code is Gauss-Hermite quadrature; needs new weight-space maths, so split into its own CIP)
- [x] Student-t on `DeepGP` (already accepted as `likelihood='student_t'` after CIP-003A; the gap listed in section 2 was out of date)
- [ ] Notebook workarounds removed; full-protocol run

## References

- `deepgp-mlp/notebooks/common.py` (`fit_sparse_gp`, `fit_deepgp`) and `notebooks/README.md`
- CIP-0028, CIP-0037 (weight-space inducing), CIP-003A (likelihood zoo)
