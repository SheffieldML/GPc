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
