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
