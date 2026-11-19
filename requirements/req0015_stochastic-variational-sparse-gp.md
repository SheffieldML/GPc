---
id: "0015"
title: "Sparse GPs can be trained by stochastic variational inference on minibatches"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
related_tenets: ["assert-and-verify", "compose-through-interfaces", "ecosystem-interoperability", "flexibility-with-performance", "lapack-owns-linear-algebra"]
stakeholders: ["maintainers", "modellers", "GPy users", "teachers"]
tags: ["svgp", "svi", "sparse", "inducing", "natural-gradients", "minibatch", "classification"]
---

# REQ-0015: Sparse GPs can be trained by stochastic variational inference on minibatches

## Description

GPc offers a sparse variational GP with an **explicit** Gaussian posterior over inducing outputs, \(q(\mathbf{u})=\mathcal{N}(\mathbf{m},\mathbf{S})\), whose evidence lower bound decomposes over data points (Hensman et al. 2013). The bound can be optimised on minibatches, so cost per step does not grow with \(N\), and it accepts non-Gaussian likelihoods through \(\mathbb{E}_{q(f)}[\log p(y\mid f)]\) (Hensman et al. 2015). Variational parameters of \(q(\mathbf{u})\) are updated by natural gradients; kernel hyperparameters, inducing inputs and likelihood parameters by an ordinary stochastic optimiser.

**Why this matters**: Today every GPc sparse approximation (DTC, FITC, PITC, DTCVAR) is collapsed and full batch, and every optimiser in `COptimisable` is a deterministic line-search or quasi-Newton method. GPc therefore cannot train on data that does not fit one batch, and its only route to non-Gaussian sparse models is EM via PPA. SVGP is the standard construction in GPy and GPflow; offering it with a verified link to the existing DTCVAR bound keeps GPc interoperable and testable.

**Who benefits**: modellers with large \(N\); classification users who want a sparse alternative to PPA/IVM with a principled bound; teachers comparing collapsed and uncollapsed bounds; the Python façade ([REQ-000E](./req000E_gpy-facing-python-facade.md)) once exposed there.

## Acceptance Criteria

- [ ] A sparse variational GP holds explicit \(q(\mathbf{u}_j)\) per output column with a shared kernel and inducing inputs, and evaluates the ELBO on the full data or as an unbiased minibatch estimate
- [ ] \(q(\mathbf{u})\) is updated by natural-gradient steps; with Gaussian noise, the full batch and step size 1, one step reaches the collapsed optimum and the ELBO equals `CGp` DTCVAR `logLikelihood` to round-off
- [ ] The implementation stays a lower bound when \(\mathbf{K}_{uu}\) is numerically singular (e.g. \(M=N\), \(\mathbf{Z}=\mathbf{X}\), long lengthscale)
- [ ] Kernel hyperparameters, inducing inputs and Gaussian noise variance have analytic gradients that pass finite-difference checks
- [ ] A stochastic optimiser (at least Adam) runs on seeded minibatches; runs are reproducible from the seed; outer-iteration progress uses the observer surface of [REQ-000F](./req000F_interceptable-optimiser-model-display.md)
- [ ] At least Gaussian and probit likelihoods are supported via \(\mathbb{E}_{q(f)}[\log p(y\mid f)]\) and its derivatives (closed form or Gauss–Hermite)
- [ ] Predictive latent mean/variance and likelihood-level predictions are available; predictions agree with `CGp` DTCVAR at the Gaussian full-batch optimum
- [ ] Parity check against a GPy `SVGP` fixture (ELBO and predictions at fixed parameters)
- [ ] Non-goals are explicit: deep GPs, multi-node / MPI / GPU, non-log-concave likelihoods (Student-t, NCNM) in the first delivery, inducing-point selection heuristics beyond fixed / random / k-means, HMC over \(q(\mathbf{u})\), changing existing collapsed `CGp` paths or defaults

## Notes (Optional)

HOW is [CIP-002A](../cip/cip002A.md). Several earlier CIPs/REQs list SVI/SVGP as a non-goal of *their* scope (CIP-0015/0017/0018, REQ-0009/000B/000C/000E); this requirement is the owner. Interacts with [REQ-0014](./req0014_observation-noise-in-likelihood.md) (noise in the likelihood, not a White kernel) and with sparse PPA ([CIP-0026](../cip/cip0026.md)) as an alternative sparse classification route.

## References

- **Related Tenets**: assert-and-verify, compose-through-interfaces, ecosystem-interoperability, flexibility-with-performance, lapack-owns-linear-algebra
- **External Links**: Hensman, Fusi & Lawrence (2013), *Gaussian Processes for Big Data*; Hensman, Matthews & Ghahramani (2015), *Scalable Variational Gaussian Process Classification*; Salimbeni, Eleftheriadis & Hensman (2018), *Natural Gradients in Practice*

## Progress Updates

### 2026-11-19
Requirement drafted as Proposed alongside CIP-002A and a Phase 1 prototype (`CSvgp`, `testSvgp`) on branch `prototype-svgp`.
