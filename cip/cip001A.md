---
author: "Neil Lawrence"
created: "2026-11-17"
id: "001A"
last_updated: "2026-11-17"
status: "Implemented"
compressed: false
related_requirements: ["0002"]
related_cips: ["0019", "0013"]
tags:
- cip
- testing
- numerical-stability
- metrics
- ci
- diagnostics
title: "Categorised numerical stability metrics and baseline trend (qig-style)"
---

# CIP-001A: Categorised numerical stability metrics and baseline trend (qig-style)

> CIPs describe HOW to achieve requirements (WHAT). This CIP extends [CIP-0019](cip0019.md)'s adversarial suite from pass/xfail **outcomes** to continuous **stability diagnostics**, following the taxonomy developed for qig ([CIP-0004](https://github.com/lawrennd/qig-code/blob/main/cip/cip0004.md) / `tests/tolerance_framework.py` / REQ-0003 numerical transparency).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Keep CIP-0019's outcome gate. Add a parallel **metric stream**: each probe emits a measured value, a category with a scientifically justified budget, and a cell label (OS / BLAS / toolchain). Persist `metrics.json` on `main` from the pinned CI cell; on PRs, compare head vs baseline and sticky-comment **forward / backward** by category — Codecov-style trend for stability itself, not only cliff-crossing.

## Motivation

CIP-0019 answers: *did we hit NaN / break an invariance / clear an xfail?* That is necessary and cheap to gate.

It does **not** answer: *is cancellation getting worse while still under budget?* A PR can inflate `pdinv` residual at cond = 1e10, or translation ΔK at offset 1e9, without flipping a single `OUTCOME`. Coverage has Codecov for that continuous story; numerics currently have only a green/red cliff.

In **qig**, CIP-0004 classified checks by *error source* (machine algebra → FD validation → integration → physical claims) and asserted against derived budgets; REQ-0003 required exposing residual / conditioning / solver labels on ill-conditioned work. GPc should import that discipline for GP linear algebra, kernels, and models — without copying quantum-specific categories.

## Detailed Description

### Relationship to CIP-0019

| Layer | CIP-0019 (done) | CIP-001A (this) |
|-------|-----------------|-----------------|
| Signal | `OUTCOME` pass/xfail/fail | `METRIC` value + budget + category |
| Trend | outcome compare (improvements / regressions) | Δvalue vs baseline; forward if residual shrinks |
| Gate | fail on unexpected fail / XPASS | soft warn by default; optional hard fail if `value > budget` or Δ exceeds threshold |
| Hardware | one Ubuntu cell sufficient | **must** pin and label the baseline cell |

Do **not** merge metrics into `run-matio-tests`. Extend `numerical_stability/` only.

### Category map (qig → GPc)

qig Categories A–G stay as inspiration. GPc uses GP-shaped names with the same idea: budget tracks the dominant error mechanism.

| GPc category | Rough qig peer | Budget (initial) | Typical probes |
|--------------|----------------|------------------|----------------|
| `A_machine` | A | rtol 1e-14 / atol 1e-15 | `logDet` on scaled I; Cholesky reconstruct on well-conditioned SPD |
| `B_linalg` | B | rtol 1e-12 / atol 1e-13 | `pdinv` residual vs cond; `jitChol` relative residual on rank-one×scale |
| `C_special` | C | rtol 1e-10 / atol 1e-12 | `lnCumGaussian` vs `log(Φ)` on moderate z; mills asymptote relative error |
| `D_analytic` | D | rtol 1e-8 / atol 1e-9 | kernel K symmetry / diag consistency; exact-vs-FITC LL when expected close |
| `D_fd` | D_numerical | rtol 1e-5 / atol 1e-5 | FD vs analytic hyperparameter grads (same 1/ε² rounding story as qig) |
| `E_invariance` | — (GP-specific) | atol 1e-8 (modest) … 1e-6 (large offset) | translation ΔK; input-rescaling LL invariance |
| `F_model` | F | rtol 1e-4 / atol 1e-5 | sparse inducing≈X LL gap; predictive variance floor |

Budgets are **starting points** documented with a one-line error-source note (as in qig), not aspirational machine precision. Tighten only with analysis; never hide a bug by loosening (qig CIP-0004 policy).

### Metric record schema

Machine-readable line (alongside existing `OUTCOME`):

```text
METRIC name=<id> category=<cat> value=<float> budget=<float> unit=<rel|abs> cell=<label>
```

JSON artifact `numerical_stability/metrics.json`:

```json
{
  "cell": {
    "os": "Linux",
    "blas": "openblas",
    "cxx": "g++",
    "runner": "ubuntu-latest"
  },
  "git_sha": "...",
  "metrics": {
    "pdinv_resid_cond_1e10": {
      "category": "B_linalg",
      "value": 2.1e-6,
      "budget": 1.0e-4,
      "unit": "abs"
    }
  }
}
```

`report.py` gains a metrics section: per-category table of value / budget / utilisation (`value/budget`).

### Baseline store and PR trend (Codecov analogue)

Codecov: upload coverage per commit → compare PR head to base commit upload.  
Here: upload **metrics.json** from the pinned cell on every green `main` run of the numerical-stability workflow.

**Proposed store (Accept choice):**

1. **Preferred (simple):** GitHub Actions artifact named `numerical-stability-metrics-main`, retained ≥90 days; PR job downloads the latest successful `main` artifact for the **same cell label**.
2. **Alternative:** orphan branch / `gh-pages` snapshot if artifact retention is too short.

PR sticky comment (extend CIP-0019 comment):

```text
## Numerical stability trend (cell: ubuntu-latest/openblas)

| metric | category | baseline | head | Δ | util |
|--------|----------|----------|------|---|------|
| pdinv_resid_cond_1e10 | B_linalg | 1.8e-6 | 9.2e-6 | ▲ worse | 9% |

Forward: N   Backward: M   Over budget: K
```

**Classification:**

| Condition | Label |
|-----------|--------|
| head value ≤ baseline × (1−ε) (or −δ abs) | **forward** |
| head value ≥ baseline × (1+ε) (or +δ abs) | **backward** |
| \|Δ\| small | stable |
| head value > budget | **over budget** (fail if hard gate on) |

ε/δ are category-specific noise floors so BLAS micro-noise does not spam “backward.”

### Cell pinning (hardware / software)

Continuous metrics **are** BLAS/LAPACK sensitive. Policy:

- Authoritative baseline: workflow matrix **one cell** — `ubuntu-latest`, system OpenBLAS, g++/gfortran from apt (same as current CIP-0019 job).
- Every `metrics.json` carries `cell` label; `compare` refuses cross-cell diffs unless `--force-cross-cell`.
- Local macOS/Accelerate runs may print metrics for developers but must not overwrite the CI baseline.

This mirrors Codecov’s single `flags: matio-tests` upload in `coverage.yml`, not a full OS matrix.

### Suite driver changes

In `testNumericalStability.cpp` (or a thin helper header):

- Keep emitting `OUTCOME` as today.
- For probes that already compute a residual / Δ / FD error, also `emitMetric(...)`.
- Pass/xfail logic unchanged: metric over budget may still `OUTCOME=passed` during soak if we start with soft metrics; Accept decides hard coupling.

Initial metric inventory (map 1:1 onto existing probes where possible):

1. `jitChol_well_conditioned_recon` — A_machine  
2. `pdinv_resid_cond_{1e2,1e6,1e10}` — B_linalg  
3. `logDet_scale_relerr` — A_machine  
4. `lnCumGaussian_logcdf_maxabs` — C_special  
5. `gradLnCumGaussian_tail_rel` — C_special  
6. `rbf_translation_maxabs_{1e3,1e6,1e9}` — E_invariance  
7. `input_rescale_ll_rel` — E_invariance  
8. `fd_grad_maxrel_extreme_hp` — D_fd  
9. `fitc_ZeqX_ll_rel` — F_model (today’s xfail — metric still recorded)

### Non-goals

- Replacing CIP-0019 outcomes or matio fixtures.
- Multi-OS metric matrices in v1.
- A SaaS “Numericsov” product — Actions artifacts + sticky comment are enough.
- Porting qig quantum categories B/C literally.

### Optional WHAT requirement

If Accept wants a durable principle beyond REQ-0002: add `requirements/req000D_numerical-transparency.md` (qig REQ-0003 peer): *extreme but representable probes expose measured residual/cancellation/grad error with category budgets; CI trends those metrics on a pinned cell.* Link it here before implementation.

## Implementation Plan

1. **Schema + emit helpers** in the C++ suite; document categories/budgets in `numerical_stability/README.md` and a short `docs/numerics.md` subsection.
2. **Extend `report.py`** to parse `METRIC` lines → metrics section + `metrics.json`.
3. **Baseline upload** on `main` in `.github/workflows/numerical-stability.yml`; PR job download + `compare_metrics.py` (or extend `compare.py`).
4. **Sticky comment** delta table beside the CIP-0019 outcome summary.
5. **Unit tests** for parse/classify (forward/backward/over-budget) with fixture JSON.
6. **Soak:** soft metrics only for one release cycle; then Decide hard fail on over-budget / severe backward.

## Backward Compatibility

- Existing `OUTCOME` lines and exit policy unchanged.
- Old CI without metrics still valid; metrics are additive.
- Local runs without the CI cell label simply omit baseline compare.

## Testing Strategy

- Fixture JSON: synthetic baseline/head → expected forward/backward/over-budget.
- Local: `./testNumericalStability | tee log; python3 numerical_stability/report.py log` shows metrics table.
- CI on this CIP’s PR: artifact present; comment shows Δ vs `main` once baseline exists (first `main` run after merge seeds baseline — document “no baseline yet” empty state).

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md) — verification trustworthy (extends with a continuous numerics signal).
- Optional new REQ (numerical transparency) — create on Accept if desired.

## Implementation Status

- [x] Categories + budgets documented; emit `METRIC` from suite
- [x] `report.py` / `metrics.json`
- [x] CI baseline artifact on `main` + PR compare
- [x] Sticky comment trend table
- [x] Fixture tests for metric classify
- [ ] Soak policy recorded; hard-gate decision (soft soak active; `--hard-gate` optional)

## References

- [CIP-0019](cip0019.md) — outcome suite + report surface  
- [CIP-0013](cip0013.md) — bench cell / performance (orthogonal continuous signal)  
- qig CIP-0004 — tolerance categories (`lawrennd/qig-code`)  
- qig REQ-0003 — numerical stability and transparency  
- Codecov comparing commits / `coverage.yml` upload model  
- Higham — accuracy and stability of numerical algorithms (error-source thinking behind budgets)
