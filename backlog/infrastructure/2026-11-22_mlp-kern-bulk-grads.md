---
id: "2026-11-22_mlp-kern-bulk-grads"
title: "CMlpKern: bulk getGradParams / getGradX (CIP-002D remainder)"
status: "Proposed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "infrastructure"
related_cips: ["002D", "0035", "0028"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- performance
- kernels
- mlp
---

# Task: CMlpKern bulk grads — CIP-002D remainder

## Description

CIP-002D's follow-up landed a bulk `CMlpKern::compute` (`syrk`/`gemm` + element-wise asin map). Profiling of `CDeepGp` training (composite task, B=256, M=64, hidden width 5) still spends **50–70% of step time** in the MLP kernel: `getGradParams` and `getGradX` walk each pair with its own `asin`/`sqrt`/`erf`-style work. JAX does the same maths as one matrix product plus whole-array ops. With RBF, GPc is already at or slightly ahead of the JAX prototype for L≥2; MLP stays 1.1–2× slower.

Rewrite `CMlpKern::getGradParams` and `getGradX` (and ARD siblings if they share the hot path) to reuse the bulk Gram / cross-Gram, then apply the element-wise map and reductions — the same pattern CIP-002D used for stationary kernels.

## Acceptance Criteria

- [ ] `getGradParams` / `getGradX` hot paths no longer nest per-pair scalar kernel evaluations as the dominant cost
- [ ] Agreement with the retained element oracle to `MATCHTOL` (`testKern` / existing MLP FD)
- [ ] Deep-GP composite timing (B=256, M=64, D=5, L=2–3, MLP kernel) closes most of the gap to the RBF-kernel times on the same machine
- [ ] Suites green under default checks

## Implementation Notes

- Reference timings (2-core, after `marginalsGradAt` GEMM fix): L=3 MLP 29–30 ms/step vs RBF 20–22 ms; JAX MLP 23 ms (+2.5 s compile).
- Prior art: [2026-11-20_cip002D-followup-bulk-diag-grad-inducing](./2026-11-20_cip002D-followup-bulk-diag-grad-inducing.md) (Completed for `compute` only).
- Do not change weight-space `ws*` closed forms (CIP-0028).

## Related

- CIP: 002D, 0035 (DeepGP step cost), 0028
- PR context: out of scope for #142; measured alongside that work

## Progress Updates

### 2026-11-22

Task created from DeepGP vs JAX timing notes. Gap left after #141 / #142 GEMM `marginalsGradAt` fix.
