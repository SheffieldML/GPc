---
id: "lapack-owns-linear-algebra"
title: "LAPACK Owns the Linear Algebra"
status: "Active"
created: "2026-11-13"
last_reviewed: "2026-11-13"
review_frequency: "Annual"
conflicts_with: []
tags:
  - tenet
  - numerics
  - lapack
  - blas
  - performance
---

## Tenet: lapack-owns-linear-algebra

**Title**: LAPACK Owns the Linear Algebra

**Description**: Dense linear algebra is a solved problem for this codebase. `CMatrix` is a deliberate thin interface over LAPACK and BLAS, not a general-purpose matrix library seeking feature parity with Eigen or MATLAB. Architecture-specific `make.*` files exist so builds can pick tuned BLAS. We own GP modelling, transforms, and optimisation orchestration; we do not reimplement Cholesky, GEMM, or eigensolvers unless LAPACK cannot express what we need. When numerical work appears in a pull request, the default question is which BLAS/LAPACK call should carry it.

**Quote**: *"Own the model; borrow the linear algebra."*

**Examples**:
- Implementing matrix products and factorisations through LAPACK/BLAS wrappers in `CMatrix`
- Selecting ATLAS, Accelerate, or vendor BLAS via `make.inc` rather than rewriting kernels in C++
- Keeping FORTRAN numerical routines behind `ndlfortran` where that is the established path
- Spending optimisation effort on calling sequences and memory layout, not on hand-rolled GEMM

**Counter-examples**:
- Writing a custom Cholesky "just for this one GP approximation" when LAPACK already provides it
- Pulling in a second dense matrix stack that duplicates `CMatrix` responsibilities without a clear interoperability plan
- Ignoring architecture-tuned BLAS in favour of naive triple loops for production paths
- Treating `CMatrix` as a place to accumulate unrelated application logic

**Conflicts**:
- Rare conflict with "Compose Through Interfaces" if a new matrix backend is proposed behind an abstraction
- Resolution: Any alternate backend must preserve the `CMatrix` calling patterns used by kernels and GPs, and must still bottom out on battle-tested numerical libraries

**Version**: 1.0 (2026-11-13)
