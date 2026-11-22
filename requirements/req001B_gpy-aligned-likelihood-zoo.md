---
id: "001B"
title: "GPy-aligned observation likelihoods are available in C++ and via the gpc façade"
status: "Proposed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces", "assert-and-verify", "flexibility-with-performance"]
stakeholders: ["GPy users", "modellers", "maintainers", "teachers"]
tags: ["likelihood", "noise", "gpy", "python", "facade", "skew", "student-t", "poisson", "parity"]
---

# REQ-001B: GPy-aligned observation likelihoods are available in C++ and via the gpc façade

## Description

Modellers who know GPy's likelihood catalogue can use the **same observation families** in GPc: each family is a `CNoise` implementation in C++ (parameters, log-density / site oracles, hyperparameter gradients) and a thin **`gpc.likelihoods`** façade that wires into existing models (`GPRegression`, `GPClassification`, SVGP / DeepGP where the inference path already supports non-Gaussian likelihoods).

The first delivery closes the documented "broader likelihood zoo" gap left after [REQ-0014](./req0014_observation-noise-in-likelihood.md) / [REQ-0019](./req0019_nonconjugate-inference-ep-laplace.md): **Student-t**, **Poisson**, **Binomial**, **Exponential**, **Gamma**, and the recent **SkewGaussian** / **SkewExponential** families from GPy CIP-0014. Existing GPc-only families (ordered categorical, NCNM, scale) remain; survival likelihoods (Weibull, LogLogistic, LogGaussian) and mixed / multioutput likelihood containers may follow in a later slice.

**Why this matters.** Interoperate With the GP Ecosystem means speaking the likelihood names GPy users already use, including the Ferrari / O'Hagan skew models just landed in GPy. Compose Through Interfaces requires new families as `CNoise` / `ndlnoise` oracles, not forks of `CGp`. Assert Invariants and Verify Against Reference demands gradient checks and GPy / SciPy fixtures before shipping. Flexibility Without a Performance Tax keeps quadrature and element work in C++; Python stays a façade.

**Who benefits.** GPy users porting scripts; lap-time and robust-regression modellers needing skew / Student-t; teachers demonstrating count and robust observation models.

## Acceptance Criteria

- [ ] Documented GPy ↔ GPc likelihood matrix lists Match / Gap / Defer for every exported GPy likelihood and every GPc-only noise type
- [ ] C++ provides `CNoise` subclasses (or equivalent typed implementations) for at least: Student-t, Poisson, Binomial, Exponential, Gamma, SkewGaussian, SkewExponential, with stream `type` strings and `readNoiseFromStream` registration
- [ ] Each new family implements the CIP-0027 / CIP-0033 oracle surface needed by its inference path: `logLikelihood` / link log-density and derivatives for Laplace; EP moments or documented quadrature fallback; `expectedLogLikelihood` where SVGP / DeepGP will use the family
- [ ] Inverse Mills ratio (and derivatives) for skew live as shared C++ helpers, verified against GPy `gaussian_special` / SciPy
- [ ] Python package exposes `gpc.likelihoods` constructors with GPy-familiar names and parameter paths (`likelihood.variance`, `likelihood.alpha`, …); models accept a likelihood object or documented string/type
- [ ] Automated checks compare densities and gradients to GPy-generated or SciPy reference fixtures within documented tolerances; `checkGradients` passes for each new `CNoise`
- [ ] Non-goals for the first delivery are explicit: survival likelihoods (Weibull, LogLogistic, LogGaussian); `MixedNoise` / `MultioutputLikelihood` containers; heteroscedastic Gaussian as a first-class GP; warped GP; full GPy link-function zoo beyond identity / probit / log as needed per family; putting Python in the likelihood hot path

## Notes (Optional)

HOW is [CIP-003A](../cip/cip003A.md). Builds on [REQ-0014](./req0014_observation-noise-in-likelihood.md), [REQ-0019](./req0019_nonconjugate-inference-ep-laplace.md), [REQ-000E](./req000E_gpy-facing-python-facade.md). Extends the former non-goal language in REQ-000E / REQ-0009 / CIP-0015 (Poisson / Student-t deferred from the first tutorial subset). Mirror design for skew: GPy [CIP-0014](https://github.com/SheffieldML/GPy) (`GPy/likelihoods/skewed.py`).

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces, assert-and-verify, flexibility-with-performance
- **External Links**: GPy `likelihoods/`; O'Hagan & Leonard (1976); Azzalini (1985); SciPy `skewnorm`

## Progress Updates

### 2026-11-22
Requirement drafted as Proposed alongside CIP-003A after auditing GPc `CNoise` vs GPy `likelihoods` (including SkewGaussian / SkewExponential on GPy branch `backport/stage3-and-skew`).
