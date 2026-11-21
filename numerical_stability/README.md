# Numerical stability suite (CIP-0019 + CIP-001A)

Adversarial / extreme-input checks for GPc, **separate** from the matio
reference track (`make run-matio-tests`). Contracts mirror GPy CIP-0011:
finiteness, accuracy, and invariance under extreme but representable inputs.

Known defects are reported as `xfailed`. An unexpected pass (`xpassed`) fails
the job until the marker is removed (strict xfail policy).

CIP-001A adds a parallel **metric stream**: measured residuals / gaps with
category budgets and a pinned-cell baseline trend (Codecov-style for numerics).

## Run

```bash
# Reference / unit track (unchanged)
make USE_MATIO=1 matio-tests && make run-matio-tests

# This suite only
make run-numerical-stability

# Or
make testNumericalStability && ./testNumericalStability
```

Optional cell / sparse-algebra labels (CI sets `ubuntu-openblas`):

```bash
export GPC_NS_CELL=local
export GPC_NS_BLAS=accelerate   # or openblas, etc.
export GPC_SPARSE_LINALG=stable # or legacy — CIP-0012 path for sparse CGp
```

The suite appends the mode to the metric cell (`local+stable` / `local+legacy`).
Sparse probes cover DTC, DTCVAR, FITC, and PITC. Each also emits an in-process
Legacy↔StableFactor LL/grad gap (`*_legacy_stable_*`) so a single run already
shows dual-path agreement; a dual-cell run compares the full metric surface.

**Improvement experiments** (always both modes, ignore `GPC_SPARSE_LINALG`):

| Probe | What it checks |
|-------|----------------|
| `*_crowdZ_*` / `*_protocol_*` | Stable finite; Stable FD not worse than Legacy; if Legacy throws → `*_legacy_breaks_stable_ok` |
| `sparse_gamma_M_dual_path_slice` | Reduced γ×M dual-path LL/grad gaps (DTCVAR+FITC) |

Mild GPs often agree to ~1e-12 (Stable **is** the default). Stress cells are where
StableFactor can win — better FD under crowded inducing, or Legacy nonfinite.

```bash
# StableFactor vs Legacy: two suite runs + compare_metrics trend
python3 numerical_stability/compare_sparse_modes.py
# → numerical_stability/mode_compare/{stable,legacy}/metrics.json
# → numerical_stability/mode_compare/trend.md
```

Each case prints machine-readable lines:

```text
OUTCOME name=<id> result=passed|failed|xfailed|xpassed
METRIC name=<id> category=<cat> value=<float> budget=<float> unit=<rel|abs> cell=<label>
```

Exit code is non-zero on `failed` or `xpassed`. Metric over-budget is **soft**
during soak (does not fail the suite alone).

**Sticky PR reports** often show ⚠️ over-budget rows with **Status: known debt**
and **New: 0**. That means standing soft metrics re-measured after a
path-triggered run — not that the PR introduced them. Catalogue:
[KNOWN_DEBT.md](./KNOWN_DEBT.md).

## Categories (CIP-001A)

Metric budgets follow the CIP-001A **error-source** table — they are **not** the
OUTCOME pass/fail cliffs (those stay looser on purpose).

| Category | Metric budget | Error source |
|----------|---------------|--------------|
| `A_machine` | rtol 1e-14 / atol 1e-15 | fp roundoff |
| `B_linalg` | `max(1e-13, n κ ε)` | LAPACK residual |
| `C_special` | rtol 1e-10 (log abs); mills `O(1/z²)` → 1e-4 | special fn / asymptote |
| `D_analytic` | rtol 1e-8 / atol 1e-9 | analytic identity |
| `D_fd` | 1e-5; extreme HP derived 1e-3 | FD step + landscape κ |
| `E_invariance` | atol 1e-8 … 1e-6 | modest → large shared offset |
| `F_model` | rtol 1e-4 | model approximation gap |

Named constants live next to `emitMetric` in `testNumericalStability.cpp`.

## Report (Codecov-style)

```bash
make run-numerical-stability 2>&1 | tee numerical_stability/suite.log
python3 numerical_stability/report.py numerical_stability/suite.log
```

Writes `report.md`, `outcomes.json`, and `metrics.json`.

On GitHub Actions the workflow also:

1. Writes the markdown into the **job summary**
2. Uploads report / outcomes / metrics / log as `numerical-stability-report`
3. On **main**: uploads `numerical-stability-metrics-main` (90-day baseline)
4. On **PRs**: downloads that baseline, runs `compare_metrics.py`, appends a
   forward/backward trend table, and sticky-comments the combined report

## Metric trend

```bash
python3 numerical_stability/compare_metrics.py \
  baseline/metrics.json numerical_stability/metrics.json \
  --md numerical_stability/trend.md
```

| Flag | Meaning |
|------|---------|
| `--force-cross-cell` | Allow compare when cell labels differ |
| `--hard-gate` | Exit 1 on over-budget or backward (default: soft) |

Cross-cell compare is refused by default so macOS/Accelerate local runs never
overwrite or falsely trend against the Ubuntu OpenBLAS baseline.

## Cross-branch outcome comparison

Same suite source → two trees → outcome diff:

```bash
python3 numerical_stability/compare.py \
  --baseline main \
  --candidate HEAD \
  --suite-from candidate \
  --report numerical_stability/last-compare.md
```

**Exit code:** non-zero on outcome regressions (and on suite drift if `--strict-suite`).

## Layer coverage

1. Linear algebra (`jitChol`, `pdinv`, `logDet`)
2. Special functions (`ndlutil` Gaussians / gamma)
3. Kernels (stationary + lin/white/bias/mlp/compound; translation)
4. Models (`CGp` FTC / DTC / FITC; probit noise)

## See also

- [KNOWN_DEBT.md](./KNOWN_DEBT.md) — standing xfails + soft over-budget metrics
- [CIP-0019](../cip/cip0019.md) — outcome suite + report surface
- [CIP-001A](../cip/cip001A.md) — categorised metrics + baseline trend
- GPy CIP-0011 — sibling policy in the GPy repo
- GPc CIP-0011 — unrelated (`CGp.cpp` coverage)
