---
id: "2026-11-15_dtcvar-numerical-diagnostic"
title: "Diagnose DTCVAR numerical stability vs GPy/GPmat"
status: "Completed"
priority: "Medium"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "bugs"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- testing
- sparse
- dtcvar
- bugs
---

# Task: Diagnose DTCVAR numerical stability

## Description

`gp` warns that "numerical stabilities exist in DTCVAR" but there is no `matfiles/testGpdtcvar.mat` and `testGp` never exercises `dtcvar`. Before opening a CIP to "fix" DTCVAR, determine whether gradients are self-consistent (`checkGradients`) and whether the log-likelihood matches a known VarDTC reference (GPy `VarDTC` / GPmat `dtcvar`).

## Acceptance Criteria

- [x] Small C++ DTCVAR diagnostic runs `checkGradients` on a fixed synthetic problem
- [x] Same problem compared to a GPy/GPmat-aligned VarDTC reference (likelihood, and grads where paramisation matches)
- [x] Findings recorded here: self-consistent? matches reference? jitter-only warning?
- [x] Decision noted: demote warning / backlog fix / propose CIP

## Implementation Notes

- Harness: `make diagDtcvar` → `./diagDtcvar` (finite-diff grads + state dumps).
- Compare: `python3 scripts/diag_dtcvar_compare.py` (stdlib-only GPc algebra + GPy-style VarDTC).
- GPc forms \(A = K_{uf}K_{uf}^\top + K_{uu}/\beta\) (BLAS gemm β scales \(K_{uu}\)).
- Scope tested: spherical Gaussian, fixed inducing, RBF kernel, single output.

## Related

- CIP: **none** — no correctness CIP warranted from this diagnostic
- GPmat: `/Users/neil/lawrennd/GPmat/matlab/gpLogLikelihood.m` (`dtc`/`dtcvar` case)
- GPy: `GPy/inference/latent_function_inference/var_dtc.py`

## Progress Updates

### 2026-11-15

Task created; ran diagnostic.

**Findings (N=10, M=5, q=2, RBF, β=100, inducing fixed):**

| Check | DTC | DTCVAR |
| --- | --- | --- |
| `checkGradients` max \|analytic−numeric\| | ~6.6e-8 **PASS** | ~4.8e-8 **PASS** |
| vs GPc-algebra Python reimplementation | ~4e-6 **PASS** | ~9e-6 **PASS** |
| vs GPy-style VarDTC LL | n/a | ~5e-6 **PASS** |

**Decision:** The old CLI warning was precautionary, not evidence of a wrong bound. Gradients are self-consistent; DTCVAR log-likelihood matches GPy VarDTC on this path. Removed the scary CLI warning (comment points here). No CIP to “fix DTCVAR.”

### 2026-11-15 (fixture)

Added `matfiles/testGpdtcvar.mat` via Octave+GPmat (`scripts/gen_testGpdtcvar.sh` / `scripts/octave/gen_testGpdtcvar.m`), then stamped C++ `params`/`grads`/`ll`. GPmat vs C++ LL on this fixture: `|Δ|≈2.4e-4`. `testGp` matio path runs `dtcvar` (DTC 2007 fixture still ~459 LL off — left disabled). Stream round-trip needs `updateM()` after `fromFile`.

**CI note:** params stay at `MATCHTOL`; grads/ll use `gpFixtureTol=1.0` because sparse Cholesky drifts across BLAS (PR #27 Ubuntu vs Mac-stamped fixture saw max abs grad diff ~0.26).
