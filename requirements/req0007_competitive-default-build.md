---
id: "0007"
title: "Default Linux build and kernel evaluations stay within stated factors of a tuned reference"
status: "In Progress"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
related_tenets: ["flexibility-with-performance", "lapack-owns-linear-algebra", "assert-and-verify", "compose-through-interfaces"]
stakeholders: ["maintainers", "users building from source"]
tags: ["performance", "build", "kernels", "benchmark"]
---

# REQ-0007: Default Linux build and kernel evaluations stay within stated factors of a tuned reference

## Description

A maintainer who follows the documented Linux instructions gets a tuned BLAS and an optimised binary. The reference-test build keeps its bounds checks. An in-repo benchmark reports the wall time of one objective-plus-gradient evaluation for a full GP, variational DTC, and FITC, on data and parameters shared with an independent reference. A timing is published only when the log-likelihoods agree, apart from constants the project has already recorded.

On the published two-dimensional spherical RBF protocol, with one BLAS thread, the full GP is no slower than the optimised OpenBLAS build measured on 2026-11-15, and variational DTC and FITC are at most 1.3× the independent reference. The benchmark record includes input dimension, so a q=2 result is not read as a claim about higher-dimensional inputs.

**Why this matters**: Flexibility Without a Performance Tax is empty if the default Linux build links reference BLAS with no optimisation level. LAPACK Owns the Linear Algebra means that default should be a tuned BLAS. Assert Invariants and Verify Against Reference means a timing is withheld when the likelihood is wrong. Compose Through Interfaces means a faster kernel path still runs for compound kernels, including the RBF-plus-white model used in the comparison.

**Who benefits**: Anyone following the Linux build instructions; maintainers who need a before/after number when changing a kernel or a sparse approximation.

REQ-0001 and REQ-0002 are already validated. They require that the tools build and that the reference suite runs. They do not require a tuned BLAS, an optimised default, or a timing gate. This requirement is that further outcome.

## Acceptance Criteria

- [ ] The documented Linux build links a tuned BLAS and compiles with optimisation (`-O2` or equivalent, bounds checks off). On the published protocol its one-evaluation time is no worse than the 2026-11-15 OpenBLAS `-O2 -DNDEBUG` measurements. The reference-test configuration keeps bounds checks
- [ ] An in-repo benchmark reports the median time of one objective-plus-gradient evaluation for the full GP, variational DTC, and FITC, on data shared with an independent reference
- [ ] The benchmark withholds a timing unless the log-likelihoods agree with that reference, allowing only constants the project has already recorded
- [ ] On the q=2, one-thread protocol the full GP does not regress, and variational DTC and FITC are at most 1.3× the independent reference
- [ ] Each benchmark record includes input dimension, BLAS library, compiler flags, and thread count
- [ ] Compound kernels, including RBF plus white noise, take the fast kernel path, and that path agrees with the element-wise implementation to the project's match tolerance, including near-duplicate inputs

## Notes (Optional)

The 1.3× sparse factor is the acceptance bar for this requirement. A tighter sparse ratio after the whitened sparse path is a re-benchmark owned by CIP-0012, using this protocol. It is not an acceptance criterion here.

The 2026-11-15 table compares OpenBLAS with reference BLAS under `-O2 -DNDEBUG`. Shipped `make.linux` is `-ggdb -DDBG` with no optimisation level and `-llapack -lblas`. `-DDBG` changes method visibility in `CGp.h`. It does not add numeric checks. The optimised-versus-unoptimised gap is part of meeting the first criterion, and it had not been measured when this requirement was written.

## References

- **Related Tenets**: flexibility-with-performance, lapack-owns-linear-algebra, assert-and-verify, compose-through-interfaces
- **Builds on**: [REQ-0001](req0001_contemporary-build.md) (tools build), [REQ-0002](req0002_runnable-verification.md) (reference suite runs)
- **HOW**: [CIP-0013](../cip/cip0013.md)

## Progress Updates

### 2026-11-15
Requirement added so CIP-0013 implements a performance outcome. REQ-0001 and REQ-0002 stay validated for build and reference tests.

### 2026-11-16
CIP-0013 Accepted → In Progress. P1 (OpenBLAS + `RELEASE=1`) and P2 (`GPC_CHECKS`) landed; GPc `benchGPc` driver in tree. GPy compare gate and P3–P5 still open.
