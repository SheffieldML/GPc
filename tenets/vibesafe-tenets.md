# Gpc Tenets

*Generated on 2026-11-13*

This document combines all individual tenet files from the project.

---
id: "assert-and-verify"
title: "Assert Invariants and Verify Against Reference"
status: "Active"
created: "2026-11-13"
last_reviewed: "2026-11-13"
review_frequency: "Annual"
conflicts_with: ["flexibility-with-performance"]
tags:
  - tenet
  - testing
  - correctness
  - numerics
---

## Tenet: assert-and-verify

**Title**: Assert Invariants and Verify Against Reference

**Description**: Gaussian process code fails quietly when shapes, symmetries, or gradients drift. GPc treats dimension and bound checks as part of the design language (`DIMENSIONMATCH`, `BOUNDCHECK`, `MATRIXPROPERTIES`, `SANITYCHECK`) and pairs them with tests that compare C++ behaviour to reference artefacts under `matfiles/`. Optimisable models expose `checkGradients`. When you change a kernel, noise model, or sparse approximation, you owe both local assertions at API boundaries and a verification path against a known-good result. Correctness bugs in likelihood gradients are more expensive than an assert that fires early.

Today those reference tests exist in-tree but are not a working default path: the makefile notes that compiling `tests` requires `_NDLMATLAB`, and there is no CI. Restoring a runnable verification suite is part of making the current software trustworthy; decoupling that suite from MATLAB can wait for a later release.

**Quote**: *"If the shapes disagree, stop; if the gradients disagree, do not ship."*

**Examples**:
- Guarding kernel `compute` paths with dimension and symmetry checks before writing into `K`
- Running `testKern`, `testNoise`, and friends against reference fixtures under `matfiles/`
- Calling `checkGradients` after implementing a new `COptimisable` objective
- Using `equals` tolerances (`ndlutil::MATCHTOL`) when comparing matrices and models after I/O round-trips
- Preferring an assert on active-set size over silently truncating an index vector
- Treating "the tests compile and run on this machine" as a release gate, even while fixtures remain MATLAB-era `.mat` files

**Counter-examples**:
- Catching assertion failures and continuing with a default matrix size
- Landing a new approximation type with no gradient check and no reference comparison
- Shipping platform or build changes without proving `make` and the test binaries still work
- Removing `DIMENSIONMATCH` calls because they "clutter" a hot path without proving the invariant elsewhere
- Updating stream readers without a test that old and new parameterisations still match

**Conflicts**:
- Can conflict with "Flexibility Without a Performance Tax" if asserts dominate profiled hot loops in release builds
- Resolution: Keep rich checks in debug and test builds; retain cheap, high-value invariants (sizes, symmetry flags) where failures would corrupt results
- Can conflict with "Interoperate With the GP Ecosystem" while verification still depends on MATLAB libraries
- Resolution: Make the current MATLAB-gated path work first; replace the fixture and loader stack in a later release

**Version**: 1.1 (2026-11-13)


---
id: "compose-through-interfaces"
title: "Compose Through Interfaces"
status: "Active"
created: "2026-11-13"
last_reviewed: "2026-11-13"
review_frequency: "Annual"
conflicts_with: ["flexibility-with-performance"]
tags:
  - tenet
  - design
  - interfaces
  - modularity
---

## Tenet: compose-through-interfaces

**Title**: Compose Through Interfaces

**Description**: GPc builds models by composing small contracts rather than by forking monolithic solvers. Kernels implement `CKern`, noise models implement `CNoise`, trainable objects implement `COptimisable`, and persistence goes through `CStreamInterface` and (optionally) `CMatInterface`. New covariance functions, likelihoods, or optimisers should arrive as implementations of those contracts. Compound structure is first-class: `CCmpndKern` and tensor kernels assemble behaviour from parts. The GP, GPLVM, IVM, and PPA entry points consume these interfaces; they should not hard-code a single kernel or noise type.

**Quote**: *"Extend by fulfilling the contract, not by rewriting the model."*

**Examples**:
- Adding a Matérn or ARD kernel as a `CKern` subclass without touching `CGp` training logic
- Swapping Gaussian, probit, ordered, or ncnm noise through the shared `CNoise` API
- Forming an RBF-plus-bias-plus-white kernel with `CCmpndKern` instead of a bespoke combined class
- Driving scaled conjugate gradients through `COptimisable` so GP, noise, and other models share one optimiser surface
- Serialising any streamable component via `writeParamsToStream` / `readParamsFromStream`

**Counter-examples**:
- Special-casing RBF behaviour inside `CGp` instead of going through `CKern`
- Copy-pasting SCG into a new model rather than implementing `getOptParams` / `computeObjectiveGradParams`
- Introducing a new likelihood that bypasses `CNoise` and talks only to one executable
- Encoding kernel type as a switch statement scattered across prediction, gradient, and I/O code

**Conflicts**:
- Can conflict with "Flexibility Without a Performance Tax" if interface layers force extra heap traffic in hot loops
- Resolution: Keep pure virtual boundaries coarse; allow concrete helpers and precomputed state inside each implementation

**Version**: 1.0 (2026-11-13)


---
id: "ecosystem-interoperability"
title: "Interoperate With the GP Ecosystem"
status: "Active"
created: "2026-11-13"
last_reviewed: "2026-11-13"
review_frequency: "Annual"
conflicts_with: ["flexibility-with-performance"]
tags:
  - tenet
  - interoperability
  - formats
  - usability
  - legacy
---

## Tenet: ecosystem-interoperability

**Title**: Interoperate With the GP Ecosystem

**Description**: GPc is one node in a wider Gaussian process toolchain. Users should move data and models across that toolchain without bespoke converters for every experiment. The durable commitment is shared, inspectable formats: plain-text `.model` streams people can read, and SVM-light-style data shared with other SheffieldML GP software. Portability across compilers and OSes (gcc, MSVC, architecture make files) is part of the same idea: meet users where their scientific stack already is.

MATLAB bridges (`CMatInterface`, mex, `.mat` fixtures) were the original interoperability path and still explain much of the test and visualisation design. They also date the setup. Near-term work is to keep the current stack working and verified, including whatever MATLAB-gated paths still unlock today’s fixtures. Dropping or replacing those bridges belongs in a later release, with an explicit migration story so users and fixtures are not stranded.

**Quote**: *"Speak the formats the community already uses."*

**Examples**:
- Loading tutorial data from SVM-light files shared with other SheffieldML GP software
- Writing human-readable `.model` files via `CStreamInterface` versioned key-value streams
- Keeping stream field names versioned so old `.model` files remain loadable
- Retaining MSVC and multi-architecture build paths so interoperability is not Linux-only
- Getting the present build and MATLAB-dependent verification working before deciding what to drop
- Planning mex/`CMatInterface` retirement as a future-release CIP once a replacement verification path exists

**Counter-examples**:
- Inventing a binary-only checkpoint format with no text escape hatch
- Changing stream field names without a version bump, breaking old `.model` files
- Deleting MATLAB bridges in the middle of a "make it work again" effort, leaving tests unable to run
- Treating mex maintenance as sacred merely because it shipped in 2005
- Using a data format that only one private preprocessing script understands

**Conflicts**:
- Can conflict with "Flexibility Without a Performance Tax" when I/O bridges or text formats add cost
- Resolution: Keep interoperability at the boundaries (load, save, tests, optional language bridges); never in the SCG inner loop
- Can conflict with maintenance cost when legacy MATLAB tooling bit-rots
- Resolution: Stabilize and verify first; retire MATLAB bridges in a later release once durable formats and a non-MATLAB verification path are in place

**Version**: 1.2 (2026-11-13)


---
id: "flexibility-with-performance"
title: "Flexibility Without a Performance Tax"
status: "Active"
created: "2026-11-13"
last_reviewed: "2026-11-13"
review_frequency: "Annual"
conflicts_with: ["compose-through-interfaces"]
tags:
  - tenet
  - design
  - performance
  - architecture
---

## Tenet: flexibility-with-performance

**Title**: Flexibility Without a Performance Tax

**Description**: GPc exists because MATLAB GP toolboxes forced a choice between speed and model flexibility. We write the modelling layer in C++ so users can compose kernels, noise models, and sparse approximations without paying the interpreter tax that made flexible MATLAB designs slow. Performance is not an afterthought bolted on later; it is a constraint on every design choice. When a new abstraction would make the code clearer but would force unnecessary allocations, copies, or Python/MATLAB round-trips in the inner loop, we reject it or push the cost outside the hot path.

**Quote**: *"Be flexible in the model, thrifty in the inner loop."*

**Examples**:
- Choosing C++ so compound kernels and interchangeable noise models remain practical at training scale
- Routing matrix algebra through LAPACK and BLAS rather than reimplementing dense linear algebra in C++
- Offering FTC, DTC, FITC, and PITC approximations so large data sets stay tractable without abandoning the GP interface
- Prefetching distances and kernel submatrices (`updateX`) when X changes, instead of recomputing them on every element access
- Keeping command-line training of the oil and sinc examples interactive (minutes, not hours) on ordinary hardware

**Counter-examples**:
- Adding a high-level scripting layer that reallocates full kernel matrices on every likelihood evaluation
- Preferring an elegant object graph that copies `CMatrix` objects in the SCG inner loop
- Implementing a new sparse approximation that looks clean in code but is asymptotically worse than FITC for the intended data size
- Porting hot paths to a slower host language for developer convenience

**Conflicts**:
- Can conflict with "Compose Through Interfaces" when deeper polymorphism adds virtual-call or allocation overhead
- Resolution: Keep interfaces at model boundaries (kernel, noise, optimiser); keep element-wise and BLAS-bound work concrete and allocation-light

**Version**: 1.0 (2026-11-13)


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
