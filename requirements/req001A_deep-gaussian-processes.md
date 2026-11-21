---
id: "001A"
title: "Deep Gaussian processes can be trained on minibatches with any inducing family per layer"
status: "Proposed"
priority: "Medium"
created: "2026-11-21"
last_updated: "2026-11-21"
related_tenets: ["assert-and-verify", "compose-through-interfaces", "flexibility-with-performance", "lapack-owns-linear-algebra"]
stakeholders: ["maintainers", "modellers", "GPy users", "teachers"]
tags: ["deep-gp", "dsvi", "svi", "inducing", "weight-space", "minibatch"]
---

# REQ-001A: Deep Gaussian processes can be trained on minibatches with any inducing family per layer

## Description

Users can fit a deep GP — a stack of GP layers with mean functions — by doubly stochastic variational inference on minibatches, choosing the kernel and the `CInducing` family per layer, with Gaussian, probit or ordered likelihoods on the output.

**Why this matters.** Compositional data (functions of learned low-dimensional projections, sharp transitions) are out of reach for single-layer GPs. In the CIP-0035 prototype a three-layer deep GP reaches test RMSE 0.23 where the exact one-layer GP reaches 0.83 and a tuned MLP of the same shape 0.34. CIP-0030 delivered the explicit-input SVGP core intended as the layer surface; weight-space MLP features (CIP-0028) also need a deep setting to be judged.

**Who benefits.** Modellers with compositional or non-stationary data; GPy / GPflow users looking for deep GPs in GPc; teachers comparing deep GPs with networks of the same shape.

## Acceptance Criteria

- [ ] A deep GP with \(L\ge1\) layers optimises an unbiased minibatch estimate of the DSVI ELBO, with reparameterised per-point sampling between layers and a seeded sampler
- [ ] Each layer can use inducing points or weight-space features; a one-layer deep GP equals `CSvgp`'s ELBO, and a two-layer model with an RBF/points final layer matches the exact expectation from psi statistics within Monte Carlo error
- [ ] Gradients for every layer's kernel, inducing and variational parameters and the likelihood pass finite-difference checks at fixed random numbers
- [ ] Depth defaults (identity mean functions, per-layer variance scaled with depth) keep the prior end-to-end Jacobian spread bounded in depth, checked by a diagnostic utility
- [ ] Gaussian and probit/ordered output likelihoods are supported
- [ ] Predictions return mixture predictive densities with latent and predictive moments
- [ ] Non-goals are explicit: deep GP-LVM, natural gradients for hidden layers, learned mean functions, sampling-based (SGHMC) posteriors, GPU backends, changes to existing model defaults

## Notes (Optional)

HOW is [CIP-0035](../cip/cip0035.md). Builds on [REQ-0015](./req0015_stochastic-variational-sparse-gp.md) (SVGP) and [REQ-0018](./req0018_bayesian-gplvm-without-psi-statistics.md) (explicit-input core); REQ-0018 lists deep GPs as its non-goal, and this requirement owns them.

## References

- **Related Tenets**: assert-and-verify, compose-through-interfaces, flexibility-with-performance, lapack-owns-linear-algebra
- **External Links**: Damianou & Lawrence (2013), *Deep Gaussian Processes*; Salimbeni & Deisenroth (2017), *Doubly Stochastic Variational Inference for Deep Gaussian Processes*

## Progress Updates

### 2026-11-21
Requirement drafted as Proposed alongside CIP-0035.
