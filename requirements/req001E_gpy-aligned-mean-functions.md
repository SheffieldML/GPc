---
id: "001E"
title: "GP models can have a parametric prior mean function, matching GPy's mapping-based mean functions"
status: "Proposed"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
related_tenets: ["assert-and-verify", "compose-through-interfaces", "ecosystem-interoperability", "flexibility-with-performance"]
stakeholders: ["maintainers", "modellers", "GPy users", "teachers"]
tags: ["mean-function", "gpy", "mappings", "regression", "sparse", "svgp"]
---

# REQ-001E: GP models can have a parametric prior mean function, matching GPy's mapping-based mean functions

## Description

Users can give a GP model a prior mean function \(\mu(\mathbf x)\), so that \(f\sim\mathcal{GP}(\mu, k)\), where \(\mu\) is a parametric mapping whose parameters are trained jointly with the kernel and noise hyperparameters. The catalogue of mappings, and the observable behaviour of a model that uses one, match GPy's `GPy.mappings` and its `mean_function=` argument: constant, linear, identity, additive and compound combinations, kernel-expansion, MLP and piecewise-linear.

**Why this matters.** GPc today has only a per-output constant offset and scale (`bias`, `scale`) and, in the deep GP, a fixed linear map. A GPy user who writes `GPy.models.GPRegression(X, Y, mean_function=Linear(1, 1))` has no equivalent. Without one, the GP must explain trends — linear drift, an offset that varies with an input — with the kernel, which inflates length scales and degrades extrapolation. Mean functions are also the mechanism behind skip connections in deep GPs, so a shared interface lets the deep GP's fixed maps and user-facing mean functions be one concept.

**Who benefits.** Modellers who want a parametric trend plus a GP residual; GPy users porting scripts; tutorial authors reproducing GPy's mean-function demonstrations; deep GP users who want learned skip paths.

## Acceptance Criteria

- [ ] A model-agnostic mean function contract (value, parameter gradients, input gradients, streaming, parameter paths) that new mappings implement without touching model code
- [ ] Constant, linear, identity, additive and compound mappings in the first delivery; kernel-expansion, MLP and piecewise-linear mappings in a later delivery
- [ ] Dense (FTC) GP regression supports a mean function; the likelihood, predictions and every hyperparameter gradient pass finite-difference checks (`checkGradients`) and agree with GPy on a shared fixture within `ndlutil::MATCHTOL`
- [ ] Sparse regression approximations that GPy supports with a mean function (variational/DTC-var) match GPy; GPc's remaining approximations either support it with a derived and checked gradient or refuse it with a typed error that names the approximation
- [ ] SVGP supports a mean function at both data and inducing inputs, as GPy's does
- [ ] Predictions (latent and predictive means, samples) include the mean function; predictive variances are unchanged by it
- [ ] Mean function parameters are reachable by GPy-style paths (REQ-0008), can be fixed and constrained, and survive a `.model` stream round-trip; models written before this change load unchanged
- [ ] A model without a mean function is bit-identical in objective, gradient and prediction to the model before this change
- [ ] Combinations that are not supported (heteroscedastic noise, uncertain inputs, non-Gaussian inference paths not yet derived) fail loudly, never silently ignore the mean function
- [ ] Non-goals are explicit: sparse spectral and warped GPs; mean functions on the GPLVM latent space in the first delivery; GPU backends; replacing the `bias` / `scale` output transforms

## Notes (Optional)

HOW is [CIP-0042](../cip/cip0042.md). GPy's own coverage is uneven — its Laplace, FITC, DTC and PEP inference assert that no mean function is present — so "match GPy" means match where GPy implements it and do the derivation, with a gradient check, where GPc goes further.

## References

- **Related Tenets**: assert-and-verify, compose-through-interfaces, ecosystem-interoperability, flexibility-with-performance
- **External Links**: GPy `GPy/mappings/`, `GPy/core/gp.py` (`mean_function`), `GPy/inference/latent_function_inference/exact_gaussian_inference.py`, `svgp.py`
- Related requirements: [REQ-0008](./req0008_gpy-like-parameter-access.md), [REQ-001A](./req001A_deep-gaussian-processes.md), [REQ-000A](./req000A_shared-model-interchange.md)

## Progress Updates

### 2026-11-28
Requirement drafted with status Proposed. It lifts the "mean functions" non-goal recorded in [REQ-0009](./req0009_gpy-tutorial-parity-native.md), [REQ-000E](./req000E_gpy-facing-python-facade.md) and [CIP-0015](../cip/cip0015.md).
