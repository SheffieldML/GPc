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
