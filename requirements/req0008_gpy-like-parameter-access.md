---
id: "0008"
title: "Users can set and constrain parameters by GPy-like paths"
status: "Completed"
priority: "Medium"
created: "2026-11-16"
last_updated: "2026-11-16"
related_tenets: ["compose-through-interfaces", "ecosystem-interoperability", "assert-and-verify", "flexibility-with-performance"]
stakeholders: ["maintainers", "modellers"]
tags: ["parameters", "gpy", "api", "constraints", "naming"]
---

# REQ-0008: Users can set and constrain parameters by GPy-like paths

## Description

GPc exposes a documented path-based parameter API so callers can set values and (where supported) constraints without memorising flat integer indices. **Canonical path and leaf names match GPy/`paramz` where the parameterisation allows** (e.g. `rbf.lengthscale`, `rbf_1.variance`, `mlp.weight_variance`). Where GPc stores a different quantity (RBF `inverseWidth`), the façade converts so get/set uses GPy’s natural units. Indexed `setParam` / `getParam` remain the native storage API.

**Why this matters**: Index-only setting in `gp.cpp` and friends is brittle across kernel types and compound layouts. Interoperate With the GP Ecosystem means a user who writes `kern.rbf.lengthscale = 0.2` in GPy should find the same path and units here. Compose Through Interfaces keeps this as a façade over `CKern` / `CNoise` / `CGp`, not a second parameter system in the optimiser loop.

**Who benefits**: Anyone building models in C++ or demos; maintainers of CIP-000D (fixing coregionalization entries); users porting GPy scripts.

## Acceptance Criteria

- [x] Documented path grammar resolves unique parameters on single and compound kernels using GPy-style names (and model-level fields where implemented)
- [x] Duplicate components follow paramz-style `name`, `name_1`, `name_2`, …
- [x] Get/set by path uses GPy natural units, with documented conversions where storage differs (at least iso RBF `lengthscale` ↔ `inverseWidth`)
- [x] Legacy GPc leaf names remain accepted as aliases; `paramPaths()` lists GPy canonical names
- [x] Callers can apply or replace positivity / unit-interval constraints by path without rewriting constructors
- [x] Ambiguous or missing paths fail loudly (assert or typed error); no silent truncation
- [x] Indexed APIs and stream I/O remain valid; old `.model` files still load
- [x] Reference or unit tests cover path resolution, unit conversion round-trips, and constraint changes
- [x] Non-goals are explicit for the first delivery: full GPy `Parameterized` tree, regex `m['.*']`, tied parameters, pretending RBF-ARD `inputScale` is GPy `lengthscale_i`, and rich priors; per-parameter `fix`/`unfix` may be phased

## Notes (Optional)

HOW was [CIP-0014](../cip/cip0014.md) (Closed). Durable usage notes: [docs/param-paths.md](../docs/param-paths.md). Stream persistence of fix masks / non-default transforms remains deferred.

## References

- **Related Tenets**: compose-through-interfaces, ecosystem-interoperability, assert-and-verify, flexibility-with-performance
- **External Links**: GPy `Parameterized`, `Param`, `constrain_positive`, `fix`

## Progress Updates

### 2026-11-16
Requirement created with CIP-0014 sketch.

### 2026-11-16 (naming)
Canonical paths and units locked to GPy/`paramz` where sensible; legacy GPc names are aliases; RBF-ARD `inputScale` explicitly not aliased to `lengthscale_i`.

### 2026-11-16 (accept)
CIP-0014 Accepted; requirement moved to In Progress.

### 2026-11-16 (complete)
Phases A–C merged (PRs #44, #45); CIP-0014 Closed and compressed; requirement Completed.
