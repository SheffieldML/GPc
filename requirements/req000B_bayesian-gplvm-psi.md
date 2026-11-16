---
id: "000B"
title: "Users can train a Bayesian GP-LVM with psi statistics (GPy-parity subset)"
status: "In Progress"
priority: "Medium"
created: "2026-11-16"
last_updated: "2026-11-16"
related_tenets: ["compose-through-interfaces", "assert-and-verify", "flexibility-with-performance", "ecosystem-interoperability", "lapack-owns-linear-algebra"]
stakeholders: ["maintainers", "modellers"]
tags: ["gplvm", "bayesian", "psi", "uncertain-inputs", "variational", "gpy"]
---

# REQ-000B: Users can train a Bayesian GP-LVM with psi statistics (GPy-parity subset)

## Description

GPc provides a variational Bayesian GP-LVM path comparable to GPy’s `BayesianGPLVM` for the common case: diagonal Gaussian posterior over latent inputs \(q(\mathbf{X})\), inducing-point sparse approximation, and closed-form psi statistics (\(\psi_0, \psi_1, \psi_2\)) for the kernels that admit them (at least RBF / RBF-ARD, Linear, Bias, White, and additive compounds of those).

Existing MAP `CGplvm` remains available. Uncertain-input expectations live on the kernel contract so sparse GPs and the Bayesian GP-LVM share one implementation rather than forking model-specific kernel maths.

**Why this matters**: Compose Through Interfaces says psi belongs on `CKern`, not inside one model class. Assert Invariants requires shape checks and gradient verification against known-good references. Flexibility Without a Performance Tax forbids allocating full \(N \times M \times M\) tensors when the ELBO only needs \(\sum_n \Psi_2^{(n)}\). Interoperate With the GP Ecosystem means a modeller who knows GPy BayesianGPLVM should recognise the GPc path. LAPACK Owns the Linear Algebra keeps the inducing-side factorisation on the StableFactor / DTCVAR algebra already in flight for sparse GPs.

**Who benefits**: Users of latent-variable visualisation and dimensionality reduction who need regularised \(q(\mathbf{X})\) rather than MAP point estimates; maintainers closing a documented GPy gap that CIP-0015 / REQ-0009 explicitly deferred.

## Acceptance Criteria

- [ ] Callers can form diagonal Gaussian \(q(\mathbf{X})\) (mean and positive variance per latent dimension) and optimise its parameters jointly with kernel, inducing inputs, and noise precision
- [ ] Psi statistics \(\psi_0, \psi_1, \psi_2\) (and gradients needed for training) exist as a `CKern` contract for at least RBF/RBF-ARD, Linear, Bias, and White; compounds compose correctly
- [ ] A trainable Bayesian GP-LVM model exposes a collapsed Titsias-style ELBO plus \(\mathrm{KL}(q(\mathbf{X})\|p(\mathbf{X}))\), with `checkGradients` green on a small problem
- [ ] Verification artefacts compare psi values and/or ELBO (and selected gradients) to an independent reference (GPy-generated text fixtures and/or Monte Carlo), without requiring MATLAB at runtime
- [ ] Stream or documented save/load round-trips \(\boldsymbol{\mu}\), \(\mathbf{S}\), inducing inputs, and kernel parameters
- [ ] Non-goals are explicit for the first delivery: MRD, SSGPLVM, SVI/SVGP, HMC, dynamics / GPDM priors on \(q(\mathbf{X})\), Matérn closed-form psi, full tensor-product psi, EP/Laplace, Python bindings

## Notes (Optional)

HOW is [CIP-0017](../cip/cip0017.md). Amends the non-goal lists in [REQ-0009](./req0009_gpy-tutorial-parity-native.md) and [CIP-0015](../cip/cip0015.md) once Accepted: Bayesian GPLVM moves from “future CIP” to this requirement. Builds on MAP [CGplvm](../CGplvm.h), DTCVAR / inducing machinery in `CGp`, and the whitened sparse algebra of [CIP-0012](../cip/cip0012.md).

## References

- **Related Tenets**: compose-through-interfaces, assert-and-verify, flexibility-with-performance, ecosystem-interoperability, lapack-owns-linear-algebra
- **External Links**: Titsias & Lawrence (2010), *Bayesian Gaussian Process Latent Variable Model*; GPy `models.bayesian_gplvm`, `kern.src.psi_*`

## Progress Updates

### 2026-11-16
Requirement created with CIP-0017 (Proposed). Feasibility review: no psi or \(q(\mathbf{X})\) in tree today; MAP GPLVM and DTCVAR are the reusable base.
