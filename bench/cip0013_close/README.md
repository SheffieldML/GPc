# CIP-0013 close re-benchmark (2026-11-21)

q=2 spherical RBF protocol from [cip0013.md](../../cip/cip0013.md), one BLAS thread (`OPENBLAS_NUM_THREADS=1`), `make RELEASE=1` `./benchGPc`, gated by `bench/run_bench.py` (GPc / GPy / NumPy LL agreement).

Machine: Apple Silicon (arm64), OpenBLAS via Homebrew, g++-16 `-O2 -DNDEBUG`, `GPC_CHECKS=0`.

All published rows: `gate=pass` and `gpc_over_gpy ≤ 1.3` (REQ-0007 sparse bar). Full GP is faster than the 2026-11-15 OpenBLAS column on the same N (no regression).

Raw: [2026-11-21_q2_rebench.csv](./2026-11-21_q2_rebench.csv).
