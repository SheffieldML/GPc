---
id: "0009"
title: "Users can recreate the documented GPy-tutorial subset via native C++ demos and markdown"
status: "In Progress"
priority: "Medium"
created: "2026-11-16"
last_updated: "2026-11-18"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces", "assert-and-verify", "flexibility-with-performance"]
stakeholders: ["maintainers", "modellers", "teachers"]
tags: ["gpy", "tutorials", "demos", "documentation", "multioutput"]
---

# REQ-0009: Users can recreate the documented GPy-tutorial subset via native C++ demos and markdown

## Description

GPc ships a documented subset of the workflows shown in SheffieldML GPy tutorial notebooks and `GPy.examples`, recreatable without Python: short compiled C++ demos, CLI recipes, and markdown tutorials that invoke those programs and embed gnuplot (or equivalent) outputs. GPy notebooks remain the **spec and checklist**, not the delivery runtime.

The first subset is the **easy and close** surface already enabled by the kernel catalogue, sparse DTCVAR path, path-based parameters, and coregionalization kernels: plain and sparse regression, compound kernels with GPy-like paths, and multi-output ICM/LCM toys. Sharing via Jupyter/Colab and Python bindings is explicitly out of scope for this requirement.

**Why this matters**: Interoperate With the GP Ecosystem means a modeller who worked through GPy’s `basic_gp`, `sparse_gp_regression`, or coregionalization tutorials should find a corresponding C++ path. Compose Through Interfaces keeps demos as thin clients of `CKern` / `CNoise` / `CGp`. Assert Invariants requires helpers and demos to be checked, not only narrative. Flexibility Without a Performance Tax rejects putting a scripting layer in the optimiser loop; demos and docs stay at the boundary.

**Who benefits**: Users porting GPy scripts; teachers who want C++ lab material parallel to GPy notebooks; maintainers closing the end-to-end demo gap left by REQ-0006.

## Acceptance Criteria

- [ ] A published checklist maps each in-scope GPy tutorial/example to a GPc demo and/or markdown section (or an explicit “not in subset” note)
- [ ] Multi-output helpers exist so callers need not hand-stack designs: at least `build_XY` (and slice helpers comparable to GPy `get_slices` / `index_to_slices`)
- [ ] A thin factory assembles ICM or LCM plus per-output noise (`indexwhite`) into a trainable `CGp` (or documented equivalent) without a new approximation type
- [ ] At least one compiled demo each for: 1D full GP regression, sparse DTCVAR regression, coregionalization toy, and compound kernel with path get/set
- [ ] Markdown tutorials invoke demos or `./gp` / related CLIs and show how to produce plots (gnuplot path is sufficient)
- [ ] CLI exposes kernel type strings needed by those demos where the types already exist in the stream/factory (e.g. Matérn / stdperiodic), without resurrecting broken FITC as a tutorial default
- [ ] Demos fail loudly on shape/index misuse (dimension and bound checks), consistent with project asserts
- [ ] Non-goals are explicit: Python bindings / Jupyter as the share format (owned by [REQ-000E](./req000E_gpy-facing-python-facade.md)), EP/Laplace classification in `CGp`, Poisson / Student-t / heteroscedastic / warped GPs, mean functions, SVI / HMC, integral kernels, FITC-as-default sparse tutorial. Bayesian GPLVM is owned by [REQ-000B](./req000B_bayesian-gplvm-psi.md) / [CIP-0017](../cip/cip0017.md); MRD by [REQ-000C](./req000C_manifold-relevance-determination.md) / [CIP-0018](../cip/cip0018.md) — not this tutorial subset.

## Notes (Optional)

HOW is [CIP-0015](../cip/cip0015.md). Builds on validated/implemented work: [REQ-0005](./req0005_gpy-aligned-kernel-coverage.md), [REQ-0006](./req0006_multioutput-covariance.md), [REQ-0008](./req0008_gpy-like-parameter-access.md). Classification demos may point at `./ivm` as the documented substitute where GPy uses EP.

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces, assert-and-verify, flexibility-with-performance
- **External Links**: [SheffieldML/notebook GPy tutorials](https://github.com/SheffieldML/notebook); GPy `examples.regression`, `util.multioutput`

## Progress Updates

### 2026-11-16
Requirement created with CIP-0015 (Proposed). Sharing environment locked to native C++ demos + markdown/CLI; notebooks are the checklist only.

### 2026-11-16 (accept / Stage 1 launch)
CIP-0015 Accepted → In Progress. First delivery cut: redo current README/`examples/*.svml` tutorials (no `build_XY` required). Phase A helpers remain Ready in parallel for the later coreg demo. Status → In Progress.

### 2026-11-18
Python-bindings non-goal now points at [REQ-000E](./req000E_gpy-facing-python-facade.md) (Proposed) instead of an unowned “future CIP”.
