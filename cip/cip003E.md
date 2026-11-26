---
author: "Neil Lawrence"
created: "2026-11-26"
id: "003E"
last_updated: "2026-11-26"
status: "Proposed"
compressed: false
related_requirements: ["0002"]
related_cips: ["002A", "003A", "0035", "0036"]
tags:
- cip
- likelihood
- noise
- multiclass
- deepgp
- svgp
title: "Multi-class likelihoods (robust-max, softmax) for SVGP and deep GP"
---

# CIP-003E: Multi-class likelihoods (robust-max, softmax) for SVGP and deep GP

> CIPs describe HOW to achieve requirements (WHAT). REQ-0002 covers runnable verification.

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Code changes complete
- [ ] Closed - Verified and merged
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add K-class likelihoods to `CNoise`: robust-max (Hernández-Lobato et al., 2011) first, softmax second. Both couple the K latent outputs of a data point, so they need a row-wise expected-log-likelihood hook that the SVGP and deep-GP final layer call instead of the per-element `expectedLogLikelihood`.

## Motivation

GPc has Gaussian, probit, ordered, and (CIP-003A) skew, Student-t, Poisson, binomial, exponential and gamma likelihoods; none handles more than two unordered classes. CIP-0036 deferred multi-class. The MNIST deep-GP experiment therefore regressed ±1 one-hot targets with a Gaussian likelihood and formed class probabilities by Monte Carlo over per-class Gaussians. That is a baseline, not a proper classifier: its NLL is not comparable with the MLP and conv-net baselines, and the Gaussian noise model has no reason to calibrate. Robust-max has an exact one-dimensional quadrature for its expectation and is the likelihood used in the GPflow multi-class models, so it is the natural first target.

## Detailed Description

### Why the existing interface is not enough

`CSvgp::expectedLogLik(y, mu, var, gMu, gVar, j)` treats each of the d outputs independently. A K-class likelihood needs, per data point i, the vector of marginals (mu_ik, var_ik), k = 1..K, and returns E_q[log p(y_i | f_i)] with gradients with respect to every mu_ik and var_ik. Marginals stay independent across k under the SVGP factorisation, which is what makes the expectation tractable.

### Interface (on `CNoise`, not a new hierarchy)

```cpp
virtual bool isRowCoupled() const { return false; }
// y: class index (or one-hot row); mu, var, gMu, gVar: length K.
virtual double expectedLogLikelihoodRow(const double* y, const double* mu,
                                        const double* var, unsigned int K,
                                        double* gMu, double* gVar) const;
```

`CSvgp::dataTerm`, `dataTermQGrads*`, `naturalGradientStep*` and `elboGrad*` branch once on `isRowCoupled()` and loop over rows instead of elements. Default `false` leaves every existing path untouched. The gamma-weighted CVI natural-gradient step needs the per-site (gMu, gVar) only, so it carries over.

### Robust-max

p(y = c | f) = (1 - eps) if c = argmax_k f_k, else eps / (K - 1), with eps a learned or fixed parameter. With f_k ~ N(mu_k, v_k) independent,

P(c = argmax) = integral N(u | mu_c, v_c) prod_{k != c} Phi((u - mu_k) / sqrt(v_k)) du,

computed by Gauss-Hermite quadrature with Q nodes at O(K Q) per point; E_q[log p] = log(1 - eps) P_c + log(eps / (K - 1)) (1 - P_c). Gradients with respect to mu and var follow from differentiating the integrand (closed form per node). K = 2 reduces to probit with a scaled bias, which gives an exact check.

### Softmax

E_q[log softmax_c(f)] has no closed form. Use S reparameterised samples with common random numbers (fixed seed per call, so finite-difference checks are meaningful), giving an unbiased, differentiable estimate. This is stochastic inside the ELBO; the likelihood must say so in its documentation and in `display`.

### Targets and prediction

Targets are accepted as integer labels in one column or as one-hot rows; the noise model normalises internally. `out(yPred, probOut, mu, var)` returns class probabilities from the same quadrature (robust-max) or sampling (softmax). The one-hot ±1 Gaussian path stays available as an explicit baseline.

### Deep GP and Python

`CDeepGp::setLikelihood` already forwards to the final layer. The final layer needs K output columns; `isRowCoupled()` tells the deep GP to skip the Gaussian beta path. The Python `gpc.likelihoods` façade (CIP-003A) gains `RobustMax(K, eps)` and `Softmax(K, samples)`.

### Tenet alignment

- *Compose through interfaces*: the extension is a `CNoise` implementation plus one hook, not a special case inside `CSvgp`.
- *Flexibility without a performance tax*: quadrature is O(K Q) per point and allocation-light; the softmax sample count is a user knob.
- *Assert and verify*: dimension asserts on K against y and mu columns; gradient checks and reference comparisons are acceptance gates below.

## Implementation Plan

1. Row-coupled hook on `CNoise`, with `CSvgp` dispatch and the default non-coupled path unchanged.
2. `CRobustMaxNoise`: quadrature, gradients, parameter eps, stream I/O (versioned fields), `display`, `out`.
3. Verification for robust-max (see Testing Strategy).
4. `CSoftmaxNoise` with common-random-number sampling.
5. Deep-GP wiring and a K-class synthetic example plus MNIST re-run, reported against the Gaussian-one-hot baseline.
6. Python façade and documentation.

Backlog tasks are created once this CIP is Accepted.

## Backward Compatibility

Existing likelihoods and models are unaffected: `isRowCoupled()` defaults to false and the per-element path is unchanged. New `.model` fields are versioned.

## Testing Strategy

- **Reference values:** P(c = argmax) against brute-force Monte Carlo (large S) and against high-order quadrature; K = 2 against `CProbitNoise` expected log-likelihood and gradients.
- **Gradients:** finite-difference check of (gMu, gVar) per point; `checkGradients` on the full SVGP objective with the new likelihood; for softmax use fixed seeds.
- **Fixtures:** a small K = 3 case with values produced by an independent implementation (GPflow `RobustMax`), stored under `matfiles/` or `testdata/`, with the generating script in the repo.
- **Invariants:** probabilities sum to one; eps in (0, 1); asserts on K mismatches.
- **Regression:** `testSvgp`, `testNoise*`, `testDeepGp` unchanged.
- **Gate:** the tests compile and run on this machine under `make`.

## Related Requirements

- REQ-0002 (runnable verification).

## Implementation Status

- [ ] Row-coupled hook and `CSvgp` dispatch
- [ ] `CRobustMaxNoise`
- [ ] Robust-max verification
- [ ] `CSoftmaxNoise`
- [ ] Deep-GP wiring and MNIST comparison
- [ ] Python façade and docs

## References

- Hernández-Lobato, Lobato and Dupont (2011), Robust multi-class Gaussian process classification.
- Hensman, Matthews and Ghahramani (2015), Scalable variational Gaussian process classification.
- CIP-0036 (multi-class deferred), CIP-003A (likelihood zoo), CIP-002A (SVGP).
