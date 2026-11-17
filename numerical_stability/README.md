# Numerical stability suite (CIP-0019)

Adversarial / extreme-input checks for GPc, **separate** from the matio
reference track (`make run-matio-tests`). Contracts mirror GPy CIP-0011:
finiteness, accuracy, and invariance under extreme but representable inputs.

Known defects are reported as `xfailed`. An unexpected pass (`xpassed`) fails
the job until the marker is removed (strict xfail policy).

## Run

```bash
# Reference / unit track (unchanged)
make USE_MATIO=1 matio-tests && make run-matio-tests

# This suite only
make run-numerical-stability

# Or
make testNumericalStability && ./testNumericalStability
```

Each case prints a machine-readable line:

```text
OUTCOME name=<id> result=passed|failed|xfailed|xpassed
```

Exit code is non-zero on `failed` or `xpassed`.

## Cross-branch comparison

Same suite source → two trees → outcome diff:

```bash
python3 numerical_stability/compare.py \
  --baseline main \
  --candidate HEAD \
  --suite-from candidate \
  --report numerical_stability/last-compare.md
```

| Flag | Meaning |
|------|---------|
| `--baseline REF` | Code under test A (default `origin/main`) |
| `--candidate REF` | Code under test B (default `HEAD`) |
| `--suite-from` | `baseline` \| `candidate` \| `PATH` to this directory |
| `--strict-suite` | Fail on suite drift (tests present on only one side) |
| `--keep-work-dir` | Keep scratch worktrees / outcome files |

**Exit code:** non-zero on regressions (and on suite drift if `--strict-suite`).
Improvements alone exit 0.

On a branch where the suite is not yet on `main`, use
`--suite-from candidate`.

## Layer coverage

1. Linear algebra (`jitChol`, `pdinv`, `logDet`)
2. Special functions (`ndlutil` Gaussians / gamma)
3. Kernels (stationary + lin/white/bias/mlp/compound; translation)
4. Models (`CGp` FTC / DTC / FITC; probit noise)

GPy-only layers (SVGP, EP moment matching, Python link stack, AST breakpoint
scan) are out of scope.

## See also

- [CIP-0019](../cip/cip0019.md) — integration + comparer
- GPy CIP-0011 — sibling policy in the GPy repo
- GPc CIP-0011 — unrelated (`CGp.cpp` coverage)
