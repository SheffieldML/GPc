---
id: "2026-11-22_cip0013-bench-cache-reuse"
title: "CIP-0013 closed on cache-reusing timings that understated GPc cost"
class: 1
owner: "Neil Lawrence"
status: "Analysed"
created: "2026-11-22"
last_updated: "2026-11-22"
incident_date: "2026-11-21"
spawned_cips: ["0013"]
spawned_backlog: ["2026-11-16_cip0013-bench-kernel-blas"]
spawned_requirements: ["0007"]
tags:
  - coe
  - performance
  - benchmarking
  - cip0013
---

# CoE: CIP-0013 closed on cache-reusing timings

> **Correction of Error** (CIP-0016). Blameless: focus on process and tooling.

## Summary

CIP-0013 and REQ-0007 were closed on 2026-11-21 using `bench/cip0013_close/` timings that reused GPc factorisation caches across repeats. The CIP protocol required parameter perturbation between repeats; the harness never did it. Honest timings show full GP ~1.5–2× faster than GPy (not 40–75×) and sparse VarDTC/FITC ~2–3× slower (missing the 1.3× bar). After fixing the harness, profiling shows the sparse gap is mostly extra arithmetic (default LQ + dual-path Legacy A/Ainv + recomputed \(K_{uf}\)), not C++ vs Python. CIP-0013 and REQ-0007 are reopened; harness fidelity and a sequenced sparse-path cleanup are the follow-ups.
## Severity

**Class**: 1

**Blast radius**: Maintainer verification and published docs (`docs/build-and-ci.md` close table). No shipped library API change; published performance claims were wrong.

## Impact

- REQ-0007 marked Validated and CIP-0013 Closed on a measurement that did not match the written protocol.
- Docs claimed sparse GPc/GPy ≤ 0.73; corrected protocol puts sparse ratios around 2–3× on a 2-core x86 check.
- Full-GP “40–75× faster” claims are not credible under an optimiser-like workload.

## Timeline

- 2026-11-15 — CIP-0013 baseline table; protocol text already says parameters are perturbed between repeats.
- 2026-11-16 — Harness lands (`benchGPc.cpp`, `run_bench.py`) without implementing that perturbation.
- 2026-11-21 — Close rebench artifact; CIP Closed; REQ-0007 → Validated; docs table published.
- 2026-11-22 — Independent check: same parameters across repeats → GPc N²-like scaling vs GPy N³; perturbing params restores optimiser-like costs; CoE opened; harness fixed; CIP/REQ reopened.

## Evidence / metrics

Published close row (cache reuse): Full GP N=4000 GPc 38 ms vs GPy 2988 ms (ratio ~0.01).

Perturbed-params check (2-core x86 Linux VM, same `run_bench.py` gate): Full GP N=4000 GPc ~1949 ms vs GPy ~3164 ms (ratio ~0.62); VarDTC 10k/200 ~2.35×; FITC 10k/200 ~2.94×. Scaling of the published GPc full-GP column was ~4.5× per N doubling (≈N²), inconsistent with Cholesky.

Same protocol, sparse-linalg mode swap (perturbed params, N=10k M=200):

| Model | GPy | GPc StableFactor (whitened chol) | GPc LQ (default) |
|---|---|---|---|
| VarDTC | 304 ms | 292 ms | 677 ms |
| FITC | 220 ms | 357 ms | 655 ms |

Code corroboration: `CGp` defaults to `SparseLinalgLQ`; whitened `updateAD` always `_whitenKuf` then still builds Legacy `A`/`Ainv` (“retained for dual-path compares”); `CRbfKern::inducingGradZ` re-`compute`s `Kuf`/`Kuu`; `CStationaryKern::compute` copies workspace element-wise via `setVal`.
## What happened

The timing loops called `computeObjectiveGradParams` / `_objective_grads` with unchanged parameters after warm-up. GPc kept factorisations via CIP-0009 invalidation-on-change; GPy recomputed. The close gate therefore measured mostly cache hits on GPc.

## Incident questions

### Detection

1. **When did you learn there was impact?**
   - 2026-11-22, when reviewing the close table scaling against FLOP expectations for exact GP.
2. **How did you learn there was impact?**
   - Human review of published ratios; confirmed by building a perturbing variant of `benchGPc` and re-running `run_bench.py`.
3. **How can we reduce the time-to-detect in half?**
   - Add a harness sanity check: GPc full-GP time must grow roughly as N³ (or at least faster than N²) across the protocol N grid, else withhold the table.

### Diagnosis

4. **What was the underlying cause of the impact?**
   - Protocol text and harness implementation drifted: perturbation was documented and used when studying *GPy* caches, but never coded into the timed loops.
5. **Was an internal activity underway during the incident?**
   - Yes — CIP-0013 close / REQ-0007 validation.
6. **How can we reduce the time-to-diagnose in half?**
   - Keep a “cache-bust on / off” harness flag and require both columns in close artifacts.

### Mitigation

7. **When did impact return to pre-incident levels (or when will it)?**
   - Docs marked withdrawn on 2026-11-22; REQ-0007 Validated and CIP Closed undone. Restored only after a Mac reference rebench with the fixed harness meets or revises the 1.3× bar.
8. **How does the owner know the system/process is properly restored?**
   - New `bench/cip0013_close/` artifact with perturbed-params loops; full-GP scaling consistent with N³; sparse ratios reported honestly; CIP status matches that evidence.
9. **How did you determine where and how to mitigate?**
   - Diff protocol metric (`setOptParams` + obj+grad, perturb between repeats) against `benchGPc.cpp` / `run_bench.py` source.
10. **How can we reduce the time-to-mitigate in half?**
    - Treat harness protocol bullets as acceptance tests (assert perturbation or assert scaling), not prose.

## 5 Whys

1. **Why** were close ratios wrong?
   - Because GPc timings reused cached factorisations across repeats.
2. **Why** did the cache stay hot?
   - Because parameters never changed between timed calls.
3. **Why** did the harness skip perturbation?
   - Because the driver timed a bare `computeObjectiveGradParams` loop and never called `setOptParams` with a new vector.
4. **Why** did review not catch that against the CIP text?
   - Because close review checked LL gates and ratio thresholds, not that the timed workload matched the protocol’s cache-busting clause (which had been studied only for GPy).
5. **Why** was that clause easy to skip?
   - Because nothing in `run_bench.py` or CI asserted N-scaling or “params must change,” so a silent measurement bug could still flip CIP status to Closed.

**Root cause(s)**: Protocol–implementation drift in the performance harness, with no automated check that the timed path defeats caches or exhibits expected complexity.

## Traceability artifacts missing or incorrect

- [ ] Missing tenet (WHY)
- [ ] Missing requirement (WHAT)
- [x] Missing CIP (HOW) — CIP-0013 existed but its close evidence was invalid
- [x] Missing backlog task (DO) — reopen / finish harness fidelity
- [ ] Incorrect/forbidden link direction
- [ ] Template/runtime drift
- [x] Other: REQ-0007 Validated and docs close table based on invalid measurement

## Follow-up outputs

### Backlog tasks

- [x] Reopen / continue [2026-11-16_cip0013-bench-kernel-blas](../backlog/infrastructure/2026-11-16_cip0013-bench-kernel-blas.md) — owner: Neil Lawrence — harness fidelity + honest rebench
- [ ] Ordered sparse-path speedups (items 2–5 below) as backlog slices under CIP-0013 / 001B / 002D — owner: Neil Lawrence

### CIPs

- [x] Reopen [CIP-0013](../cip/cip0013.md) to In Progress — owner: Neil Lawrence — harness fidelity first; close only on honest 1.3× or revised REQ
- [ ] [CIP-001B](../cip/cip001B.md) follow-up: LQ cost — make LQ opt-in *or* drop explicit `Q` (`dorglq`) and avoid paying whitening + LQ; document that the precision-form “no \(M\times N\) solve” saving was never taken on the whitened path — owner: Neil Lawrence
- [ ] [CIP-0012](../cip/cip0012.md) / 001B residual: stop building Legacy `A`/`Ainv`/`pdinv` inside whitened `updateAD` (dual-path compares move to an explicit flag or offline tool) — owner: Neil Lawrence
- [ ] [CIP-002D](../cip/cip002D.md) residual / CIP-0013: pass cached `K_uf`/`K_uu` into `inducingGradZ` (RBF recompute today) — owner: Neil Lawrence
- [ ] CIP-0013 bulk kernel: vectorised `exp` / avoid per-element `K_of_r` + `setVal` copy out of workspace — owner: Neil Lawrence

### Requirements

- [x] Move [REQ-0007](../requirements/req0007_competitive-default-build.md) off Validated until corrected evidence — owner: Neil Lawrence

### Validation tripwires

- [ ] Harness or close script fails if full-GP GPc time ratio N=4k/N=2k is below ~5 (cache-reuse fingerprint) — owner: Neil Lawrence
- [ ] Document “perturb params / include setOptParams” as a required close checklist item in CIP-0013 — owner: Neil Lawrence
- [ ] Close artifacts record `sparse_linalg` mode (lq / stable / legacy); default-mode row is the REQ-0007 gate — owner: Neil Lawrence

### Ordered fix payoff (process note)

Not C++ vs Python at these sizes — both sit in BLAS/LAPACK. Rough order:

1. **Harness:** perturb params + time `setOptParams` (measurement gate).
2. **LQ:** opt-in default *or* cut `dorglq` / redundant work (largest sparse swing; StableFactor already ≈ GPy on VarDTC).
3. **Dual-path tax:** drop Legacy `A`/`Ainv` from the hot whitened path.
4. **Reuse \(K_{uf}\)/\(K_{uu}\)** in `inducingGradZ`.
5. **Bulk kernel / vectorised `exp`.**

## Checklist

- [x] Severity class assigned
- [x] Single-threaded owner named
- [x] Summary written (stand-alone)
- [x] Impact documented
- [x] Timeline documented (required Class 1+; optional Class 0)
- [x] Incident questions answered (Detect / Diagnose / Mitigate)
- [x] 5 Whys completed (abbreviated OK for Class 0; full for Class 1+)
- [x] Missing/incorrect traceability artifacts identified
- [x] Follow-up outputs listed with owners (and created)
- [x] CoE status updated to Analysed once outputs are created
- [ ] CoE status updated to Closed once outputs are implemented
