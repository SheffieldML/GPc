---
id: "000E"
title: "GPy users can drive a documented GPc subset from Python with familiar ergonomics"
status: "In Progress"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces", "flexibility-with-performance", "assert-and-verify"]
stakeholders: ["GPy users", "modellers", "teachers", "maintainers"]
tags: ["gpy", "python", "facade", "interop", "api", "ease-of-use"]
---

# REQ-000E: GPy users can drive a documented GPc subset from Python with familiar ergonomics

## Description

GPc exposes a **light Python façade** so a modeller who already knows GPy can construct, configure, train, and predict with a **documented modelling subset** without writing C++. The façade presents GPy-like ergonomics at the boundary: NumPy arrays for data, path-based parameter access in GPy-natural units (aligned with [REQ-0008](./req0008_gpy-like-parameter-access.md)), and short recipes that mirror the native tutorial subset of [REQ-0009](./req0009_gpy-tutorial-parity-native.md).

The façade is an **ease-of-use and sharing surface**, not a second modelling engine. Optimisation, kernel evaluation, and sparse algebra remain in the C++ core; Python must not sit in the SCG / likelihood hot path. Native C++ demos, CLI, and markdown tutorials stay first-class; this requirement does not replace [REQ-0009](./req0009_gpy-tutorial-parity-native.md). Model-file interchange with GPy remains [REQ-000A](./req000A_shared-model-interchange.md).

**Why this matters**: Interoperate With the GP Ecosystem means meeting GPy users where they already work (Python notebooks and scripts) without forcing a full rewrite into C++. Compose Through Interfaces keeps the façade a thin client of `CKern` / `CNoise` / `CGp` (and related entry points), not a fork of the model. Flexibility Without a Performance Tax forbids putting a scripting layer in the inner loop. Assert Invariants requires the façade to fail loudly on shape and path misuse and to be verified against known-good native or GPy-aligned results.

**Who benefits**: GPy users trying GPc for speed or sparse paths; teachers who want notebook-side labs that still call GPc; maintainers who need a low-friction way to exercise the C++ surface from Python tests.

## Acceptance Criteria

- [ ] A documented, installable Python package imports under a stable name and exposes a first modelling subset without requiring callers to write C++ or edit makefiles for ordinary use
- [ ] Callers can pass training and test inputs/outputs as NumPy arrays (or documented array-protocol equivalents) and receive predictions as NumPy arrays
- [ ] For the first subset, callers can at least: build a Gaussian regression GP (full / FTC) with a compound RBF-style kernel, set and read parameters by **GPy-like paths and units** consistent with [REQ-0008](./req0008_gpy-like-parameter-access.md), train (optimise), and predict mean/variance
- [ ] The documented subset maps explicitly to the native tutorial checklist of [REQ-0009](./req0009_gpy-tutorial-parity-native.md) (or states “not yet in Python subset”); at least one sparse DTCVAR-style and one path-based compound-kernel recipe are included when those exist natively
- [ ] Training and likelihood evaluation run in the C++ core; the façade does not reimplement kernels, noise, or the optimiser loop in Python
- [ ] Ambiguous or missing parameter paths, and shape mismatches, fail loudly with clear errors; no silent truncation of arrays or parameter vectors
- [ ] Automated checks compare façade results to the native GPc path and/or GPy-generated fixtures within documented tolerances (objective and/or predictions on a tiny seeded problem)
- [ ] Documentation shows a short “GPy → GPc façade” side-by-side for the first subset so a GPy user can port a basic script in an afternoon
- [ ] Non-goals for the first delivery are explicit: full GPy/`paramz` object graph and `__setattr__` tree; regex parameter indexing; replacing native demos/CLI as the primary share format for [REQ-0009](./req0009_gpy-tutorial-parity-native.md); file interchange (owned by [REQ-000A](./req000A_shared-model-interchange.md)); EP/Laplace, SVGP/SVI, HMC, Poisson/Student-t/warped GPs, mean functions; putting Python allocations or interpreters *inside* objective/gradient evaluation or kernel `compute`; claiming API identity with GPy. (Outer-iteration live progress is owned with [REQ-000F](./req000F_interceptable-optimiser-model-display.md), not forbidden.)

## Notes (Optional)

HOW is [CIP-001D](../cip/cip001D.md) (In Progress). Builds on [REQ-0005](./req0005_gpy-aligned-kernel-coverage.md), [REQ-0008](./req0008_gpy-like-parameter-access.md), and the native subset of [REQ-0009](./req0009_gpy-tutorial-parity-native.md). Live progress and structured repr consume [REQ-000F](./req000F_interceptable-optimiser-model-display.md) / [CIP-001E](../cip/cip001E.md). Distinct from [REQ-000A](./req000A_shared-model-interchange.md) (model files, not a live API). Binding technology locked in CIP-001D (pybind11).

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces, flexibility-with-performance, assert-and-verify
- **External Links**: GPy `models.GPRegression`, `Parameterized` / `paramz`; [SheffieldML/notebook](https://github.com/SheffieldML/notebook) tutorials as ergonomics checklist
- **Deferred from**: [CIP-0015](../cip/cip0015.md) / [REQ-0009](./req0009_gpy-tutorial-parity-native.md) (“Python bindings / Jupyter share format”)

## Progress Updates

### 2026-11-18
Requirement created as Proposed WHAT after confirming no CIP owned the light Python wrapper; deferred language in CIP-0015 / REQ-0009 pointed at a future CIP with no owner.

### 2026-11-18 (HOW)
[CIP-001D](../cip/cip001D.md) Proposed: pybind11 package `gpc`, GPy-like ergonomics without paramz clone, FTC then DTCVAR phases.

### 2026-11-18 (accept / Phase A)
CIP-001D Accepted → In Progress on worktree/branch `cip001D-python-facade`. Requirement → In Progress. Phase A scaffold started.

### 2026-11-18 (Phase A/B + layering)
Phase A green; Phase B `GPRegression` smoke green. Locked: notebooks/`pods`/matplotlib stay optional extras and examples (Phase E), not core `gpc` — same split as GPy.

### 2026-11-18 (display / progress)
Clarified non-goals: outer-iteration live progress and structured model repr are in scope via [REQ-000F](./req000F_interceptable-optimiser-model-display.md) / [CIP-001E](../cip/cip001E.md); CIP-001D Phase B2 binds them.
