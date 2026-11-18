---
id: "0013"
title: "Intra-evaluation covariance work can use BLAS and OpenMP without app-managed threads"
status: "Proposed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets: ["flexibility-with-performance", "lapack-owns-linear-algebra", "assert-and-verify"]
stakeholders: ["maintainers", "benchmark-users"]
tags: ["performance", "kernels", "blas", "openmp", "concurrency"]
---

# REQ-0013: Intra-evaluation covariance work can use BLAS and OpenMP without app-managed threads

## Description

Within a single likelihood or gradient evaluation, stationary covariance construction can exploit parallelism that GPc does **not** own as a thread pool: (1) distance and linear-algebra work expressed as BLAS (so a multi-threaded OpenBLAS / vendor BLAS can parallelise), and (2) bulk radial maps \(K(r)\) / workspace fills parallelised via OpenMP when the build enables it. Default single-thread bench semantics remain available; matio / element-wise kernel agreement is preserved. Same-instance concurrent API use stays forbidden under the published concurrency contract.

**Why this matters**: Flexibility Without a Performance Tax asks for speed in the inner loop without a scripting tax; LAPACK Owns the Linear Algebra says distance GEMM belongs to BLAS when \(q\) or block size warrants it; Assert and Verify means a threaded path must still match the element-wise oracle. Users who do not want to manage threads still get parallelism from the library runtime.

**Who benefits**: Maintainers chasing sparse / FTC wall time after CIP-0013 P1–P5; anyone linking OpenBLAS who can raise `OPENBLAS_NUM_THREADS` or rebuild with OpenMP.

## Acceptance Criteria

- [ ] Stationary bulk distance for large enough \(q\) (or block size) uses a GEMM-shaped form with \(r^2 \ge 0\) clamp and exact zero diagonal on symmetric blocks; small-\(q\) may keep a scalar loop
- [ ] Linear / already-BLAS kernel paths continue to use `gemm` / `syrk` (no regression to scalar loops)
- [ ] Optional OpenMP over disjoint row fills of stationary workspace / \(K(r)\) maps is available behind a documented build switch; default CI and one-thread bench protocol remain correct without requiring OpenMP
- [ ] Thread-count policy is documented: when the application (or OpenMP) already uses multiple threads, BLAS nesting defaults to one thread unless a measured override is recorded
- [ ] Bulk and element-wise \(K\), gradients, and `getGradX` agree to `MATCHTOL` (including near-duplicate points); matio reference suite stays green
- [ ] GPc does not introduce a custom thread pool or mutexed same-instance model API for this requirement

## Notes (Optional)

- Safety / multi-instance contract remains [REQ-0010](./req0010_documented-concurrency-contract.md) / [CIP-001F](../cip/cip001F.md). This requirement is L4 performance only.
- Competitive q=2 one-thread bars in [REQ-0007](./req0007_competitive-default-build.md) stay authoritative for the published protocol; multi-thread speedups are reported separately with thread counts.

## References

- **Related Tenets**: flexibility-with-performance, lapack-owns-linear-algebra, assert-and-verify
- **Related**: [CIP-0013](../cip/cip0013.md) P4 GEMM distances / P7 study; [CIP-0009](../cip/cip0009.md) caches

## Progress Updates

### 2026-11-18
Requirement created with CIP-0025 (Proposed).
