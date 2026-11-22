# CIP-0013 q=2 re-benchmark

## 2026-11-22 honest rebench (cache-busting)

Artifact: [`2026-11-22_q2_honest_rebench.csv`](./2026-11-22_q2_honest_rebench.csv).

Protocol: `bench/run_bench.py` + `make RELEASE=1 bench`, `OPENBLAS_NUM_THREADS=1`,
params perturbed every timed step (includes `setOptParams`). LL gate vs GPy
passed on every row. `sparse_linalg` recorded per row (CoE tripwire).

Machine: macOS arm64, Homebrew GCC 16 + OpenBLAS, GPy 1.15.0 (conda `py311`).

### Full GP (no sparse mode)

| N | GPc ms | GPy ms | GPc/GPy |
|---|-------:|-------:|--------:|
| 1000 | 35.3 | 52.5 | 0.67 |
| 2000 | 244 | 301 | 0.81 |
| 4000 | 1620 | 1789 | 0.91 |

Cache-reuse fingerprint: median(N=4000)/median(N=2000) ≈ **6.6** (tripwire ≥ ~5).

### Sparse — default `lq` (REQ-0007 gate)

| kind | N | M | GPc ms | GPy ms | GPc/GPy | ≤1.3× |
|------|--:|--:|-------:|-------:|--------:|:-----:|
| dtcvar | 2000 | 50 | 8.6 | 6.1 | **1.41** | no |
| fitc | 2000 | 50 | 8.8 | 8.0 | 1.11 | yes |
| dtcvar | 2000 | 200 | 68 | 28 | **2.41** | no |
| fitc | 2000 | 200 | 65 | 32 | **2.02** | no |
| dtcvar | 10000 | 50 | 45 | 32 | **1.42** | no |
| fitc | 10000 | 50 | 42 | 33 | 1.28 | yes |
| dtcvar | 10000 | 200 | 331 | 155 | **2.13** | no |
| fitc | 10000 | 200 | 351 | 167 | **2.11** | no |

**Verdict:** default LQ still misses the 1.3× bar on most VarDTC/FITC cells
after the 2026-11-22 sparse-path cleanups (no Legacy A/Ainv, lazy `dorglq`,
`K_uf` reuse, bulk `mapR2ToK`). Full GP stays ahead of GPy.

### Sparse — `stable` reference (same N=10k, M=200)

| kind | GPc ms | GPy ms | GPc/GPy |
|------|-------:|-------:|--------:|
| dtcvar | 121 | 170 | **0.72** |
| fitc | 161 | 168 | **0.96** |

StableFactor meets REQ-0007 on this cell; LQ does not. Closing 0013 on the
shipped default still requires either further LQ cost cuts or a deliberate
default-mode change (tracked in CoE / CIP-001B).

## 2026-11-21 table — withdrawn

[`2026-11-21_q2_rebench.csv`](./2026-11-21_q2_rebench.csv) timed without
parameter perturbation (factorisation cache hits). Do not use for REQ-0007.
See [CoE](../../coe/2026-11-22_cip0013-bench-cache-reuse.md).
