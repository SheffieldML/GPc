---
id: "0006"
title: "Users can model correlated multiple outputs via coregionalization"
status: "Validated"
priority: "Medium"
created: "2026-11-15"
last_updated: "2026-11-20"
related_tenets: ["compose-through-interfaces", "assert-and-verify", "ecosystem-interoperability", "flexibility-with-performance"]
stakeholders: ["maintainers", "modellers"]
tags: ["kernels", "multioutput", "coregionalize", "gpy"]
---

# REQ-0006: Users can model correlated multiple outputs via coregionalization

## Description

GPc can express intrinsic/linear coregionalization for multiple outputs: a coregionalization matrix over output indices composed with a spatial (or temporal) kernel, so correlated outputs are first-class rather than independent single-output GPs. The outcome is modelling capability aligned with GPy’s `Coregionalize` building block, not a full port of GPy’s multi-output wrapper stack.

**Why this matters**: Multi-output structure cannot be approximated by adding more stationary kernels on the same inputs. Compose Through Interfaces still applies—the coregionalization piece should be a `CKern` (or a documented composition of `CKern`s). Ecosystem interoperability means users who already use ICM/LCM in GPy have a path into GPc.

**Who benefits**: Modellers with vector-valued responses; anyone bridging GPy multi-output examples into C++ training.

## Acceptance Criteria

- [x] A documented coregionalization covariance is available through the kernel interface and can be composed with an existing spatial kernel
- [x] Output index handling and parameterisation (`W`, `kappa` / B) are documented and stream-persistent
- [x] Reference tests cover kernel values and parameter gradients against a known-good generator (e.g. GPy)
- [x] At least one end-to-end path (CLI or documented API example) trains or evaluates a multi-output model using that covariance
- [x] Non-goals are explicit: SDE multi-output, ODE/LFM, IndependentOutputs/Hierarchical wrappers, and derivative multi-output kernels may follow in later work

## Notes (Optional)

This requirement is deliberately separate from REQ-0005 so single-output kernel coverage is not blocked on multi-output design.

The HOW is locked in [CIP-000D](../cip/cip000D.md) to GPy’s own objects: `build_XY` (index in the last column), `Coregionalize`, `ICM` as a product on active columns, `LCM` as a sum of those products, and a per-output noise diagonal matching `MixedNoise`. The wrapper classes named in the non-goals stay out of scope.

## References

- **Related Tenets**: compose-through-interfaces, assert-and-verify, ecosystem-interoperability, flexibility-with-performance
- **External Links**: GPy `Coregionalize`; GPy multi-output regression examples

## Progress Updates

### 2026-11-15
Requirement created alongside CIP-000D.

### 2026-11-15 (later)
CIP-000D’s open stacking and composition choices were replaced by GPy’s `build_XY`, `ICM`, and `LCM`. Requirement text is unchanged. Status stays Proposed.

### 2026-11-16
CIP-000D implemented: `activeDims`, `coregionalize`, `indexwhite`, ICM/LCM helpers, `testCoregionalize`. Status → Implemented (validate/close after CI).

### 2026-11-21
End-to-end API demo via CIP-0015 (`demo_coregionalization_toy` / `CCoregionalizedRegression`). Acceptance criteria ticked; status remains Validated.
