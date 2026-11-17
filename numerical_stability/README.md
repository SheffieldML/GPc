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

Optional cell label (CI sets `ubuntu-openblas`):

```bash
export GPC_NS_CELL=local
export GPC_NS_BLAS=accelerate   # or openblas, etc.
```

Each case prints machine-readable lines:

```text
OUTCOME name=<id> result=passed|failed|xfailed|xpassed
METRIC name=<id> category=<cat> value=<float> budget=<float> unit=<rel|abs> cell=<label>
```

Exit code is non-zero on `failed` or `xpassed`. Metric over-budget is **soft**
during soak (does not fail the suite alone).

## Categories (CIP-001A)

| Category | Budget idea | Typical probes |
|----------|-------------|----------------|
| `A_machine` | ~1e-14 / 1e-8 rel | Cholesky reconstruct; `logDet` scales |
| `B_linalg` | cond-scaled eps | `pdinv` residual vs condition |
| `C_special` | 1e-9 … 1e-2 | `lnCumGaussian` / mills asymptote |
| `D_analytic` | 1e-8 | kernel consistency (when instrumented) |
| `D_fd` | 5e-3 rel | FD vs analytic hyperparameter grads |
| `E_invariance` | 1e-8 … 1e-6 | translation ΔK; input-rescale LL |
| `F_model` | 1e-4 rel | FITC Z=X LL gap |

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

- [CIP-0019](../cip/cip0019.md) — outcome suite + report surface
- [CIP-001A](../cip/cip001A.md) — categorised metrics + baseline trend
- GPy CIP-0011 — sibling policy in the GPy repo
- GPc CIP-0011 — unrelated (`CGp.cpp` coverage)
