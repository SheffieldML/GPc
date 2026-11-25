---
id: "001C"
title: "Users can obtain verified posterior samples of GP hyperparameters"
status: "In Progress"
priority: "Medium"
created: "2026-11-25"
last_updated: "2026-11-25"
related_tenets: ["assert-and-verify", "ecosystem-interoperability", "compose-through-interfaces", "flexibility-with-performance"]
stakeholders: ["GPy users", "modellers", "maintainers", "teachers"]
tags: ["hmc", "sampling", "hyperparameters", "priors", "gpy", "csv"]
---

# REQ-001C: Users can obtain verified posterior samples of GP hyperparameters

## Description

A modeller can draw samples from the posterior over a GP's hyperparameters, rather than only a point estimate, from both the C++ library and the `gp` command line. The samples are written as plain CSV that pandas, NumPy and MATLAB read directly, in the constrained (natural) parameter space by default.

Sampling must be safe by construction. A target that has no proper posterior (a sampled parameter with neither a prior nor a bounded transform) is refused with a message that names the offending parameters, unless the user explicitly overrides the check. The treatment of transforms must be explicit: by default the density includes the change-of-variables Jacobian, so samples follow the stated prior in the natural parameter space; a switch reproduces GPy's behaviour of sampling the transformed space without it.

**Why this matters.** Interoperate With the GP Ecosystem: GPy users expect priors on any parameter, including noise variance, and CSV output. Assert Invariants and Verify Against Reference: an improper posterior or a missing Jacobian silently produces meaningless samples, so both are checked, and the sampler is tested against analytic and grid-integrated posteriors. Compose Through Interfaces: sampling targets any model that exposes the optimisable and posterior-target contracts, not a single model class. Flexibility Without a Performance Tax: leapfrog steps reuse the existing gradient path with no extra allocations of kernel matrices.

**Who benefits.** Modellers who need hyperparameter uncertainty; GPy users porting priors; teachers demonstrating why a point estimate can mislead.

## Acceptance Criteria

- [ ] Priors can be placed on kernel parameters and on the likelihood variance, and survive model cloning
- [ ] Sampling an improper target is refused with a message naming the parameters; an explicit override exists
- [ ] The Jacobian of every transform used for sampling is verified against finite differences; the Jacobian can be switched off to match GPy
- [ ] Sampler output is verified against an analytic Gaussian and a grid-integrated GP length-scale posterior within effective-sample-size tolerances
- [ ] The `gp` command line draws samples from a saved model and writes CSV with named columns
- [ ] Non-goals for the first delivery are explicit: sparse, EP, Laplace, X-optimised and scale-learnt models; NUTS; multi-chain diagnostics; Python wrapper (later phases of the HOW)

## Notes (Optional)

HOW is [CIP-003B](../cip/cip003B.md). Extends [REQ-0002](./req0002_runnable-verification.md) style verification to a stochastic algorithm.

## References

- **Related Tenets**: assert-and-verify, ecosystem-interoperability, compose-through-interfaces, flexibility-with-performance
- **External Links**: Neal (2011) MCMC using Hamiltonian dynamics; Hoffman and Gelman (2014); GPy `inference/mcmc/hmc.py`

## Progress Updates

### 2026-11-25
Drafted retrospectively after the first four CIP-003B phases (Jacobian and improper check, noise priors, `CHmc`, CSV output and `gp sample`) were implemented and tested. Remaining phases (other models, stream-persisted noise priors, Python wrapper, tutorial) keep this In Progress.
