---
id: "0016"
title: "GPU training route is chosen against a concrete use case before a backend lands"
status: "Proposed"
priority: "Low"
created: "2026-11-19"
last_updated: "2026-11-19"
related_tenets: ["flexibility-with-performance", "lapack-owns-linear-algebra", "assert-and-verify", "ecosystem-interoperability", "compose-through-interfaces"]
stakeholders: ["maintainers", "modellers"]
tags: ["gpu", "decision-record", "performance", "jax", "pytorch"]
---

# REQ-0016: GPU training route is chosen against a concrete use case before a backend lands

## Description

Before GPc grows a GPU training path, maintainers record an explicit decision among the options already sketched in [CIP-002D](../cip/cip002D.md) Part B: (A) a native device-matrix / cuBLAS-style backend inside GPc; (B) keep GPc as the CPU reference oracle and train on GPU in JAX or PyTorch; or (C) a drop-in GPU BLAS intercept. The decision is driven by a **concrete** use case (for example deep directional-VIK GPs), not by abstract readiness, and it states which precision, residency, and CI constraints are in scope.

**Why this matters**: Part A of CIP-002D makes the CPU hot paths BLAS-shaped so any of A–C stays cheap to try. Committing to A without a use case would double adjoint maintenance and add CUDA/HIP CI that default runners cannot exercise. Option B reuses verified GPc fixtures as the oracle and keeps Python outside GPc objectives ([REQ-000E](./req000E_gpy-facing-python-facade.md)). Option C is unlikely to win at current inducing sizes because of host↔device copies.

**Who benefits**: maintainers deciding whether to invest in device code; modellers who need GPU training and need a clear place to look for the chosen route; teachers comparing GPc numerics to autodiff stacks.

## Acceptance Criteria

- [ ] A written decision (this requirement moved Ready/In Progress, or a CIP that implements it) names option A, B, or C (or a documented hybrid) and the motivating use case
- [ ] The decision lists non-goals for the first GPU delivery (e.g. default GitHub-hosted GPU CI, FP32-only Cholesky, multi-node)
- [ ] If A is chosen: a follow-on CIP covers device residency, kernel bulk APIs, adjoint parity tests, and self-hosted GPU runners
- [ ] If B is chosen: GPc remains the CPU reference; a prototype ports the closed forms of interest to JAX/PyTorch and checks against GPc fixtures to stated tolerances
- [ ] If C is chosen: measured evidence shows net win at the target \(M,N\) after transfer cost; otherwise C is rejected with that evidence attached
- [ ] No GPU training code lands in GPc under this requirement alone — Part B of CIP-002D stays a sketch until the decision is Accepted and scoped

## Notes (Optional)

HOW is deferred until Accept. Starting material: CIP-002D Part B options table and recommendation to prototype B first when a use case appears (e.g. CIP-0028 deep directional VIKs). CPU quick wins (CIP-002D Part A) are prerequisites, not a GPU commitment.

## References

- **Related Tenets**: flexibility-with-performance, lapack-owns-linear-algebra, assert-and-verify, ecosystem-interoperability, compose-through-interfaces
- **Related**: [CIP-002D](../cip/cip002D.md) Part B; [CIP-0028](../cip/cip0028.md); [REQ-000E](./req000E_gpy-facing-python-facade.md); [REQ-0010](./req0010_documented-concurrency-contract.md)

## Progress Updates

### 2026-11-19
Requirement drafted as Proposed from CIP-002D Phase 4 (QW8 / GPU decision record). No GPU implementation in scope.
