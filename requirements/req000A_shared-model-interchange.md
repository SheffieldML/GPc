---
id: "000A"
title: "Users can exchange GP models across GP toolboxes via a shared inspectable format"
status: "Proposed"
priority: "Medium"
created: "2026-11-16"
last_updated: "2026-11-25"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces", "assert-and-verify", "flexibility-with-performance"]
stakeholders: ["maintainers", "modellers", "GPy users", "GPmat users"]
tags: ["interop", "yaml", "json", "gpy", "gpmat", "serialization", "model-files"]
---

# REQ-000A: Users can exchange GP models with GPy and GPmat via a shared inspectable format

## Description

A **shared SheffieldML schema home** publishes a versioned, inspectable **model interchange** (**YAML primary**; JSON as the same tree). GPc, GPy, and (later) GPmat can load and save that format for a **supported subset**, so a modeller can reconstruct an equivalent model across toolboxes without bespoke converters.

Wire types use **portable tags inspired by GPy class strings** (e.g. `kern.rbf` from `GPy.kern.RBF`); full `GPy.*` strings may be accepted as read aliases. Parameters on the wire use GPy-natural units.

Native GPc `.model` streams remain valid. Training-data formats are out of scope here (see [REQ-0011](./req0011_csv-pods-data-access.md) for CSV/`pods` with SVM-light as legacy). The shared **model** format lives at the I/O boundary; it is not used in the optimiser inner loop.

**Why this matters**: Interoperate With the GP Ecosystem currently commits to `.model` and SVM-light, while GPy persists via `to_dict` JSON and GPmat via `.mat` deconstruct/reconstruct. Semantic alignment (kernel catalogue, CIP-0014 paths/units) is advancing, but file-level interchange is missing. Assert Invariants requires round-trip fixtures, not narrative parity alone. Flexibility Without a Performance Tax keeps parsing off the SCG path.

**Who benefits**: Users moving models across SheffieldML toolboxes; maintainers verifying GPc against GPy artefacts without MATLAB; future non-MATLAB verification paths (related to [REQ-0003](./req0003_matlab-bridge-retirement.md)).

## Acceptance Criteria

- [ ] A dedicated SheffieldML repo (working name `gp-scuderia`) publishes schema 0.1, a type registry (portable tag ↔ GPy class ↔ GPc/GPmat), and golden **YAML** fixtures before GPc claims interchange support
- [ ] Portable type tags are documented as GPy-class-inspired dotted paths; `GPy.*` class strings are read aliases where supported
- [ ] The schema covers at least: model type, kernel tree, likelihood/noise, trainable parameters in **GPy-natural units**, and optional training data (JSON Schema or equivalent validates both YAML and JSON encodings)
- [ ] GPc can **write** and **read** that format for a documented first subset (at least FTC Gaussian regression with compound RBF/bias/white-style kernels), consuming the shared home (not a GPc-only private schema)
- [ ] Golden fixtures demonstrate GPy → GPc and GPc → GPy reconstruction for that subset (likelihood / predictions agree within documented tolerances)
- [ ] Existing `.model` load/save behaviour is unchanged; old `.model` files still load
- [ ] Unsupported constructs fail loudly with typed/stream errors; no silent truncation of kernel trees or parameter vectors
- [ ] The registry covers GPyTorch, GPflow and GPJax; each is either verified by a golden round-trip or explicitly marked unsupported/provisional, and unsupported constructs raise
- [ ] GPmat support is either delivered or explicitly deferred with a migration note and external tracking link
- [ ] Non-goals for the first delivery are explicit: pickle; full GPy model zoo (EP, SVGP, Bayesian GPLVM, …); replacing `.model` as GPc’s native format; putting interchange I/O in the SCG loop; keeping the schema only inside GPc

## Notes (Optional)

HOW: [CIP-0016](../cip/cip0016.md) (Proposed). Builds on [REQ-0005](./req0005_gpy-aligned-kernel-coverage.md), [REQ-0008](./req0008_gpy-like-parameter-access.md), [CIP-0014](../cip/cip0014.md) conversion tables. Distinct from [REQ-0009](./req0009_gpy-tutorial-parity-native.md) (tutorial/demo parity, not model files).

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces, assert-and-verify, flexibility-with-performance
- **External**: GPy `Model.to_dict` / `from_dict` / `save_model`; GPmat `gpWriteResult` / `gpDeconstruct`; planned `SheffieldML/gp-scuderia`

## Progress Updates

### 2026-11-16
Requirement created alongside CIP-0016 sketch after positioning review of GPc `.model`, GPy JSON, and GPmat `.mat` dialects.

### 2026-11-16 (locks)
Shared `gp-scuderia` home first; YAML primary / JSON isomorphic; portable tags derived from GPy class strings.

### 2026-11-25
Scope widened beyond GPy/GPmat to GPyTorch, GPflow and GPJax; wire format stays framework-neutral.
