---
author: "Neil Lawrence"
created: "2026-11-26"
id: "003E"
last_updated: "2026-11-26"
status: "Accepted"
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
- [x] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Code changes complete
- [ ] Closed - Verified and merged
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add K-class likelihoods (robust-max, then softmax) to GPc by changing the likelihood contract once, across the library, instead of adding a hook to one model. `CNoise` gains a block size (coupled output columns per datum; 1 for every existing likelihood), block oracles whose defaults are the existing per-element code, and a capability set, so that each inference engine (SVGP, Laplace, EP, IVM, PPA, samplers, prediction, I/O, Python) either supports a likelihood or refuses it with a clear error.

## Motivation

GPc has Gaussian, probit, ordered, and (CIP-003A) skew, Student-t, Poisson, binomial, exponential and gamma likelihoods; none handles more than two unordered classes. CIP-0036 deferred multi-class. The MNIST deep-GP experiment therefore regressed ±1 one-hot targets with a Gaussian likelihood and formed class probabilities by Monte Carlo over per-class Gaussians. That is a baseline, not a proper classifier: its NLL is not comparable with the MLP and conv-net baselines, and the Gaussian noise model has no reason to calibrate. Robust-max has an exact one-dimensional quadrature for its expectation and is the likelihood used in the GPflow multi-class models, so it is the natural first target.

## Detailed Description

### What the whole library assumes today

Every consumer of `CNoise` assumes one likelihood term per element: observation y_ij depends on latent f_ij only. That assumption is spread across the library, not confined to SVGP:

| Consumer | How it touches the likelihood | Per-element assumption |
|---|---|---|
| `CSvgp`, `CDeepGp` final layer | `expectedLogLikelihood(y, mu, var, gMu, gVar)` | scalar marginal in, scalar gradients out |
| `CLaplaceInference` | `ndlnoise::laplacePointDerivs` (log p, d1, diagonal W, dW/df) | W is a diagonal vector |
| `CEpInference` | cavity moments, site (tau, v) as N×1 vectors | one Gaussian site per element |
| `CIvm`, `CPpa`, `CSparsePpa`, `CGp`, `CGplvm` (ADF; see `cip/cip003E/call-site-audit.md`) | `getNuG`, `updateSites`, `getGradInputs` per (i, j) | per-element site updates |
| `CHmc` / variational HMC | pointwise log-density | scalar |
| `CInterchange`, streams, `.mat` I/O, Python `gpc.likelihoods`, the type-string factory in `CNoise.cpp` | type string and parameter blocks | assumes the known zoo |

A multi-class likelihood breaks the assumption in a specific way: the K latents of one datum share one observation. A hook added to `CSvgp` would fix one row of this table and leave the others silently wrong. The design below changes the model of the likelihood once and states what each consumer does with it.

### Principle: the datum is a block, and per-element is the block of size one

A likelihood term is a function p(y_i | f_i) where f_i is a block of `blockSize()` latents drawn from the output columns. Existing likelihoods have blockSize 1. Robust-max and softmax have blockSize K, equal to the number of output columns.

Targets stay N × K one-hot (this is how the oil data is already stored and how `CNoise` stores `py`). That keeps `nProcesses`, `getOutputDim()`, bias/scale vectors, `CGp` and `CSvgp` dimension logic, and every existing stream and fixture unchanged. Integer labels are converted to one-hot at the boundary (loader, `setTarget`, Python façade), never inside inference.

### Interface: block oracle first, per-element as the default case

`CNoise` gains a small block-oracle contract. Each member has a default that reduces to the existing per-element virtuals, so no existing likelihood changes.

```cpp
// Consecutive output columns forming one coupled block (1 = independent).
virtual unsigned int blockSize() const { return 1; }

// Capabilities, so each inference can refuse clearly instead of guessing.
enum Capability { CapExpectedLogLik = 1, CapPointDerivs = 2,
                  CapCavityMoments = 4, CapPredictive = 8 };
virtual unsigned int capabilities() const;   // default: derived from what the
                                             // element-wise virtuals implement

// Block oracles (default: loop the element-wise virtuals over the block).
virtual double expectedLogLikelihoodBlock(const double* y, const double* mu,
    const double* var, double* gMu, double* gVar) const;          // VI
virtual double pointDerivsBlock(const double* y, const double* f,
    double* g, CMatrix& negHessian, CMatrix* dHdf) const;         // Laplace
virtual void cavityMomentsBlock(const double* y, const double* mu,
    const double* var, unsigned int j0,
    double* dlogZdMu, double* dlogZdVar) const;                    // EP / ADF (log Z oracle arrives with block EP, Phase 2)
virtual void   predictBlock(const double* mu, const double* var,
    double* prob) const;                                           // out()
```

At `blockSize() == 1` the code takes exactly the current paths, which are the oracle defaults. There is one code path and one regression surface, not a parallel multi-class branch.

### What each consumer does

- **`CSvgp` / `CDeepGp`:** the data term, CVI natural-gradient sites and `elboGrad*` loop over blocks and call `expectedLogLikelihoodBlock`. Sites remain per element (gMu, gVar) because q(f) is factorised over output columns.
- **`CLaplaceInference`:** needs the block Hessian. Softmax has a structured negative Hessian (diag(pi) minus pi pi^T), so `pointDerivsBlock` returns it as a small matrix and Laplace solves a block-diagonal W. Robust-max is not twice differentiable at ties and reports no `CapPointDerivs`; Laplace checks the capability and errors with a message naming the likelihood.
- **EP, ADF, IVM, PPA, sparse PPA, `CGp` sites (in scope):** these work with one Gaussian site per latent, N × K sites for K classes, which is the storage `CIvm`/`CNoise` already use for multi-output models (`updateSites(m, beta, ..., g, nu, index)` runs over output columns). Sites stay diagonal across the K latents of a datum. The likelihood supplies block cavity moments: from cavity marginals N(mu_k, v_k) it returns log Z and the derivatives dlogZ/dmu_k, dlogZ/dv_k for k = 1..K, and the existing identities give the new per-latent mean and variance (mean_k = mu_k + v_k dlogZ/dmu_k; var_k = v_k + v_k^2 (2 dlogZ/dv_k - (dlogZ/dmu_k)^2)). This is the factorised-site multi-class EP of Kim and Ghahramani (2006). Given diagonal sites, the posterior over the K latents is K independent solves, so `CEpInference` generalises from N×1 to N×K sites with K prior Gram matrices (shared or per-class kernel) and a single coupled step, the moment match. ADF in `CIvm`/`CPpa`/`CSparsePpa`/`CGp` uses the same oracle. IVM point selection sums the per-latent entropy changes (`entropyChangeAdd`) over the block; this is a documented choice, tested against single-class selection at K = 1. Likelihoods without `CapCavityMoments` are refused with an error naming both.
- **Variational HMC and samplers:** use the expected or pointwise log-density through the same oracle once `CapExpectedLogLik` or `CapPointDerivs` is declared.
- **Prediction:** `CNoise::out(yPred, probOut, mu, var)` and `likelihoods(...)` route through `predictBlock`, so `CGp`, `CSvgp`, the CLI tools and Python return class probabilities from one place.
- **I/O and interchange:** stream fields are versioned; `CInterchange` rejects unknown likelihoods with an explicit message as it does today, and gains a categorical mapping only once the JSON schema defines one.
- **Python:** `gpc.likelihoods.RobustMax(K, eps)` and `Softmax(K, samples)`; `DeepGP(..., likelihood='robustmax'|'softmax')` through the existing `wrap_deepgp` branch.
- **Factory and CLI:** the type-string factory in `CNoise.cpp` and `CClctrl` gain `robustmax` and `softmax`, so command-line tools and `.model` round-trips work.

### The two likelihoods

**Robust-max** (Hernandez-Lobato et al., 2011). p(y = c | f) is 1 - eps if c is the argmax of f, else eps / (K - 1). With independent Gaussian marginals, P(c = argmax) = integral N(u | mu_c, v_c) prod_{k != c} Phi((u - mu_k) / sqrt(v_k)) du, by Gauss-Hermite quadrature at O(K Q) per datum, with analytic gradients in mu and var. Z = (1 - eps) P_c + eps/(K - 1) (1 - P_c), so cavity moments follow from the same quadrature and its derivatives. Declares `CapExpectedLogLik | CapCavityMoments | CapPredictive`. K = 2 reduces to probit, an exact check.

**Softmax.** E_q[log softmax_c(f)] by S reparameterised samples with common random numbers (seed fixed per call, so finite differences are meaningful). Cavity moments use the same sampled estimator (K-dimensional quadrature is feasible only for small K); EP with sampled moments is stochastic, so it needs fixed seeds per call, damping, and a looser convergence test, and the CIP does not claim monotone EP convergence for softmax. Declares `CapExpectedLogLik | CapPointDerivs | CapCavityMoments | CapPredictive`. Stochastic inside the ELBO and EP; documented and shown in `display`.

### Tenet alignment

- *Compose through interfaces:* one contract on `CNoise`, with capabilities; no special case in any model.
- *Flexibility without a performance tax:* block size 1 runs the existing loops; blocks use small stack arrays, no per-datum heap traffic.
- *Assert and verify:* asserts that blockSize divides the output columns and K matches; capability refusals instead of silent wrong answers; each oracle verified against a reference and finite differences.
- *Interoperate with the ecosystem:* versioned streams, GPflow reference fixtures, one-hot targets as the existing data already uses.

## Implementation Plan

1. **Audit and contract:** list every `CNoise` call site (the table above, regenerated by grep and kept in the CIP) and add `blockSize()`, `capabilities()` and the block oracles with element-wise defaults. No behaviour change; the full existing test suite must pass unchanged.
2. **Block EP and ADF:** generalise `CEpInference` sites to N×K, ADF in `CIvm`/`CPpa`/`CSparsePpa`/`CGp` through `cavityMomentsBlock`, and IVM selection over blocks; capability refusals for likelihoods without `CapCavityMoments` or `CapPointDerivs`.
3. **SVGP and deep-GP dispatch** through the block oracle.
4. **`CRobustMaxNoise`:** quadrature, gradients, eps, versioned stream I/O, `display`, factory entry, `out`/`predictBlock`.
5. **`CSoftmaxNoise`** with common-random-number sampling and `pointDerivsBlock`; Laplace block-W support.
6. **Boundaries:** label-to-one-hot conversion in loaders and Python; `wrap_likelihood` / `wrap_deepgp`; CLI factory.
7. **Experiments:** K-class synthetic example and MNIST deep GP against the Gaussian one-hot baseline.

Backlog tasks are created once this CIP is Accepted.

## Backward Compatibility

Existing likelihoods and models are unaffected: `isRowCoupled()` defaults to false and the per-element path is unchanged. New `.model` fields are versioned.

## Testing Strategy

- **No-regression gate:** after phases 1 and 3, `testNoise`, `testNoiseBatched`, `testNoiseInLikelihood`, `testLikelihoodZoo`, `testCGp`, `testEp`, `testIvm*`, `testPpa`, `testSvgp`, `testDeepGp` and `testSviBayesGplvm` pass with identical numbers.
- **Default-oracle equivalence:** for every existing likelihood, the block oracle at blockSize 1 equals the element-wise virtuals to 1e-12.
- **Reference values:** robust-max P(c = argmax) against brute-force Monte Carlo and high-order quadrature; K = 2 against `CProbitNoise` expected log-likelihood and gradients; a K = 3 fixture from GPflow `RobustMax` under `testdata/`, with its generating script in the repo.
- **Gradients:** finite-difference check of (gMu, gVar) and of eps per datum; `checkGradients` on the full SVGP and deep-GP objective with each new likelihood; fixed seeds for softmax.
- **Block EP / ADF:** at K = 1 and blockSize 1 EP, IVM, PPA numbers are identical to current fixtures; for robust-max K = 2 EP matches probit EP (`testEp` fixtures); K = 3 EP against a GPflow/GPy-style reference or brute-force numerical tilted moments (the tilted moments themselves checked against quadrature to 1e-8); EP site and log Z gradients by finite differences; IVM block selection reduces to the single-class rule at K = 1. Softmax EP uses fixed seeds and a statistical tolerance.
- **Capability refusals:** a likelihood lacking `CapCavityMoments` is rejected by EP/ADF, and one lacking `CapPointDerivs` (robust-max) by Laplace, each with the documented error; softmax Laplace is checked against a finite-difference mode and Hessian.
- **Invariants:** class probabilities sum to one; eps in (0, 1); blockSize divides the output columns; one-hot rows have exactly one 1.
- **I/O:** `.model` round-trip with `equals` at `ndlutil::MATCHTOL`; old-stream loading unchanged.
- **Gate:** everything builds and runs under `make` on this machine.

## Related Requirements

- REQ-0002 (runnable verification).

## Implementation Status

- [x] Call-site audit and block-oracle contract (no behaviour change) — Phase 1
- [x] Block EP (N×K sites), ADF in IVM/PPA/sparse PPA/CGp, capability refusals — Phase 2. Also: GPy-convention probit mode (`setGpyMode`); `CGp` EP always via the block engine; `CGp` Laplace for D scalar-block outputs. Convention: the default probit is a step function on f+ε with the softener as White in K (IVM/ADF); GPy mode is Φ(y f) on f (σ²=1). Open: GPy mode is not serialised.
- [x] SVGP and deep-GP dispatch — Phase 3 (`expectedLogLikBatch`; `testSvgpBlock`; coupled-block FD in `testDeepGp`)
- [x] `CRobustMaxNoise` and verification — Phase 4 (`testRobustMax`; fixture `testdata/robustmax`). P(argmax) by composite Gauss–Legendre with breakpoints at the Φ transitions (fixed Gauss–Hermite is inaccurate for small variances); ε is a fixed hyperparameter; capabilities are expected log-lik, cavity moments, predictive (no Laplace). The fixture is an independent stdlib Simpson reference, not GPflow output (GPflow was unavailable). Finding: EP on non-log-concave robust-max clamps site precisions at the floor when labels conflict with the prior, and the EP gradient is then not stationary; the EP gradient check uses clean labels. Robust-max marginal likelihood is flat in kernel amplitude, so the check perturbs the lengthscale.
- [x] `CSoftmaxNoise`, Laplace block W — Phase 5 (`testSoftmax`, `testCGpSoftmaxLaplace` in `testEp`). Sampled quantities use a fixed set of S antithetic standard-normal draws per object (portable xorshift + Box–Muller, seed in the `.model`), so every estimate is a deterministic function of (mu, var) and its gradients are exact derivatives of the estimate. `cavityMomentsBlock` is the exact gradient of the sampled `logZBlock`. `CLaplaceInference::runBlock` solves the stacked (N·b) system with A = K⁻¹ + W (dense, O((N·b)³); the structured Rasmussen & Williams 3.4 form is a later optimisation); at blockSize 1 it equals the scalar solver. `dNegHessian` for blockSize b is a (b·b)×b matrix, entry (a·b+c, d) = dW_ac/df_d. `CGp::setLaplaceInference` now accepts coupled blocks (one runBlock per group of b outputs). Coverage added afterwards: softmax EP dL/dK vs FD (needs eta = 1; eta < 1 is fractional EP and its gradient differs), ADF `getNuG` on coupled blocks, and SVGP end to end with softmax and robust-max (gradients vs FD to 1e-8, full batch and minibatch). Finding: CVI on a non-log-concave likelihood (robust-max) can produce an indefinite precision target, so `CSvgp::naturalGradientStep(At)` now halves the step until q stays proper (unchanged for log-concave likelihoods). Robust-max CVI still ascends only overall (single-step ELBO drop up to 8e3 in the test), so gradient-based optimisation is the recommended route for it.
- [ ] Loaders, Python, factory/CLI
- [ ] K-class example and MNIST comparison

## References

- Hernández-Lobato, Lobato and Dupont (2011), Robust multi-class Gaussian process classification.
- Hensman, Matthews and Ghahramani (2015), Scalable variational Gaussian process classification.
- CIP-0036 (multi-class deferred), CIP-003A (likelihood zoo), CIP-002A (SVGP).
