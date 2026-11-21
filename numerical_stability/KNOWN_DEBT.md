# Numerical stability — known debt (soft metrics + xfails)

This note explains the ⚠️ rows on the CIP-0019 sticky PR report. They are **not**
automatic “this PR broke numerics” signals.

**How to read a PR comment**

| Field | Meaning |
|-------|---------|
| **Status: known debt** | Suite finished; soft metrics include standing over-budget cells |
| **OUTCOME `passed` / `xfailed`** | Hard gate (job fails on `failed` or unexpected `xpassed`) |
| **METRIC ⚠️ over budget** | Soft CIP-001A soak; does **not** fail the job alone |
| **Forward / Backward / New** | Trend vs pinned `main` baseline artifact |
| **New: 0** | No newly over-budget metric names vs that baseline |

Touching path-filtered files (`CNoise.*`, `CGp.*`, `makefile`, …) **re-runs** the
soak and re-posts the report ([`.github/workflows/numerical-stability.yml`](../.github/workflows/numerical-stability.yml)).
Editing likelihood code for CIP-0033 therefore surfaces this document’s debt even
when EP is unrelated.

## Strict xfails (hard, expected)

| Id | Cause | Owner / follow-up |
|----|--------|-------------------|
| `sparse_inducing_eq_X_near_exact` | FITC with \(Z=X\) still disagrees with FTC LL beyond the outcome cliff (~`1e-4` rel). Same phenomenon as soft metric `fitc_ZeqX_ll_relgap`. | Sparse FITC exactness; not EP |
| `exp_K_X_X2_consistent` | Exp / Ornstein–Uhlenbeck \(K(X,X')\) consistency under the suite’s stress layout (see suite log cause string). | Kernel / fixture follow-up |
| `jitChol_negative_diag_raises` | (Historically listed in [CIP-0019](../cip/cip0019.md).) Relative jitter can absorb a mild negative diagonal instead of raising. Confirm still xfailed in current suite before relying on the marker. | `CMatrix::jitChol` policy |

## Soft over-budget metrics (aspirational budgets)

Budgets follow CIP-001A **error-source** categories. Some `F_model` emits use a
**floor** as `budget=` for util% display; the OUTCOME check can still pass with a
looser cliff. Do not equate “348% util” with failure.

### Point inducing vs white-noise (`wn_ls_sweep_*`)

| Metric pattern | What it measures | Why it looks bad |
|----------------|------------------|------------------|
| `wn_ls_sweep_kappa_pt_g*` | \(\kappa(k(Z,Z))\) for **point** inducing as RBF inverse-lengthscale \(\gamma\) grows | Short lengthscale → near-singular Gram. **Intentional baseline** next to WN |
| `wn_ls_sweep_kappa_wn_g*` | \(\kappa\) of white-noise window \(K_{uu}\) | Stays moderate (~\(10^3\)–\(10^4\)); usually **in** budget |
| `wn_ls_sweep_max_kappa_point` | max of the point column | Same as point blow-up |

**Interpretation:** CIP-001C / `docs/numerics.md` — use `--inducing whitenoise` when
point \(K_{uu}\) conditioning collapses. The ⚠️ on `kappa_pt_*` documents the
disease, not a regression.

### FITC at \(Z = X\) (`fitc_ZeqX_ll_relgap`)

| Metric | Cause |
|--------|--------|
| `fitc_ZeqX_ll_relgap` | FITC is not exact FTC even when inducing inputs equal training inputs; relative LL gap vs dense stays large vs `F_model` rtol \(10^{-4}\). |

Paired with xfail `sparse_inducing_eq_X_near_exact`. Gradients / dual-path
StableFactor cells are separate and often fine.

### Crowded inducing stress (`sparse_*_crowdZ_*`)

| Metric pattern | Cause |
|----------------|--------|
| `sparse_dtcvar_crowdZ_stress_ll_absdiff` (and kin) | Legacy vs StableFactor (or related) LL gap under crowded \(Z\); aspirational abs budget can flag while both paths remain finite and FD-ordered. |

Outcome checks require Stable finite and FD not worse than Legacy; soft absdiff
can still warn.

### Weight-space ridge capture (`ws_ridge_wv_sweep_capture_*`)

| Metric | Cause |
|--------|--------|
| `ws_ridge_wv_sweep_capture_wv*` / `min_capture` | Emitted capture \(\approx 0.8\)–\(0.9\) with **metric budget label 0.25** (same number as the OUTCOME floor `capture ≥ 0.25`). Util% = value/budget → looks “300%+ over budget” even when the hard check **passes**. |

Real conditioning story for ridges: window width / offsets ([CIP-0028](../cip/cip0028.md),
guards in [CIP-0032](../cip/cip0032.md)); \(\kappa\) flat in `weightVariance` is the
pass criterion, not capture util%.

## What is *not* in this debt list (yet)

- **CIP-0033 EP** (`CEpInference`, `testEp`) — not wired into
  `testNumericalStability` as of 2026-11-21. Do not attribute soak ⚠️ rows to EP
  until an EP cell exists and trends `New > 0`.

## When to act

| Signal | Action |
|--------|--------|
| `New: 0`, Forward/Backward 0, only known rows ⚠️ | No soak action required for the PR |
| `New > 0` or Forward regression on a metric this PR touches | Investigate; do not merge blind |
| Unexpected `xpassed` | Remove/update xfail — that is a hard gate |
| Want EP watched | Add a finite/convergence EP cell + metric; update this file |

## See also

- [README.md](./README.md) — how to run / report / trend
- [docs/numerics.md](../docs/numerics.md) — inducing families and conditioning
- [CIP-0019](../cip/cip0019.md) / [CIP-001A](../cip/cip001A.md)
- [CIP-001C](../cip/cip001C.md) — white-noise inducing (point \(\kappa\) remedy)
