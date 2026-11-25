---
author: "Neil Lawrence"
created: "2026-11-25"
id: "003B"
last_updated: "2026-11-25"
status: "In Progress"
compressed: false
related_requirements: ["0002", "001C"]
related_cips: ["0015", "0017", "002A", "0033", "0035"]
tags:
- cip
- inference
- hmc
- mcmc
- sampling
- hyperparameters
- optimisable
title: "Hamiltonian Monte Carlo over COptimisable parameters"
---

# CIP-003B: Hamiltonian Monte Carlo over `COptimisable` parameters

> CIPs describe HOW to achieve requirements (WHAT). REQ-001C covers sampling.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [ ] Implemented - Code changes complete
- [ ] Closed - Verified and merged
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add a Hamiltonian Monte Carlo (HMC) sampler to GPc that draws from the posterior over the parameters of any `COptimisable` model. The sampler consumes only `getOptParams`, `setOptParams` and `computeObjectiveGradParams`, so `CGp`, `CGplvm`, `CSvgp` and `CDeepGp` gain it without model changes. The first delivery targets GP hyperparameters (kernel, noise, and inducing inputs where applicable) with the existing `CDist` priors and `CTransform` reparameterisations.

## Motivation

Today every inference route in GPc is an optimisation or a deterministic approximation. HMC is listed as a non-goal in CIP-0015, CIP-0017, CIP-0033 and the GPy-parity subset checklist, so it was deferred rather than rejected. Users who want hyperparameter uncertainty (for example, marginalising length scales on small data sets) must currently leave GPc.

The repository already provides the pieces HMC needs:

- `computeObjectiveGradParams(g)` returns the negative log-posterior and its gradient. `CKern::addPriorGrad` adds the log-prior gradient, so the objective is already a log-posterior when priors are set.
- `CTransform` maps constrained parameters (positive variances, length scales) to unconstrained space. HMC needs the unconstrained space, and the optimiser already works there.
- `checkGradients` already verifies gradients, which is the main correctness risk for HMC.

HMC reuses all of this. The cost per leapfrog step is one objective and gradient evaluation, which is what SCG already pays, so there is no new work in the inner loop (tenet: flexibility without a performance tax).

## Detailed Description

### Interface

A new `CHmc` class, independent of any particular model (tenet: compose through interfaces):

```cpp
class CHmc {
 public:
  CHmc(COptimisable& model, unsigned int seed);
  void setStepSize(double eps);
  void setNumLeapfrog(unsigned int L);
  void setMassDiag(const CMatrix& m);       // diagonal mass matrix
  void adaptStepSize(unsigned int burnin, double targetAccept = 0.8);
  bool step();                              // one HMC transition; true if accepted
  void sample(unsigned int n, CMatrix& samples, CMatrix& logPost);
  double acceptRate() const;
};
```

### Algorithm

Target density \(p(\theta) \propto \exp(-E(\theta))\), where \(E\) is the value returned by `computeObjectiveGradParams` in the optimiser's (transformed) parameter space. Each transition:

1. Draw momentum \(p \sim \mathcal{N}(0, M)\).
2. Run \(L\) leapfrog steps of size \(\epsilon\) using the model gradient.
3. Accept with probability \(\min(1, \exp(H_0 - H_1))\), where \(H = E(\theta) + \tfrac12 p^\top M^{-1} p\). On rejection, restore the original parameters through `setOptParams`.

Step size adapts by dual averaging during burn-in only. After burn-in the chain is a valid Markov chain.

### Design points to settle during review

- **Jacobian.** Sampling in transformed space targets the density of the transformed variable. For a log transform, \(\theta = \log\phi\), the target gains a \(+\log\phi\) Jacobian term relative to the density of \(\phi\). `CTransform` must expose a log-Jacobian and its gradient, or the sampler must document that it samples the transformed-space density. We propose adding the Jacobian so that user-specified priors mean what they say.
- **Priors required.** Under a flat prior on a positive parameter the posterior can be improper. `CHmc` should assert that every sampled parameter has either a prior or a bounded transform, and refuse otherwise (tenet: assert invariants).
- **Objective sign and scope.** Some models' objectives omit constants or add terms only valid at the optimum (for example, collapsed sparse bounds). Phase 1 supports exact-likelihood `CGp` (Gaussian noise, FTC); other models are enabled one at a time once the target is shown to be a proper posterior.
- **Mass matrix.** Start with diagonal. Optionally initialise from the Hessian diagonal at the MAP.
- **Reproducibility.** Seed explicitly; reuse the existing `ndlutil` random generator rather than adding a second.

### Output and I/O

Samples are written as dense CSV/TSV, matching the interoperability tenet, so they load directly into pandas or `arviz`. A sample file carries the parameter names from `getOptParamNames`. No change to the `.model` stream format is needed.

### Surfaces

- C++: `CHmc.h` / `CHmc.cpp`.
- CLI: a `gpHmc` mode, or a `--hmc` flag on the existing GP executable, following the pattern in CIP-0033.
- Python: `gpc.inference.hmc(model, ...)` in the façade, outside any inner loop (CIP-001D rules apply).

## Implementation Plan

0. **Requirement.** Draft a requirement ("users can obtain posterior samples of GP hyperparameters") and link its tenets. Pause for review.
1. **Core sampler.** `CHmc` with fixed step size, diagonal mass, leapfrog, accept/reject, and restore-on-reject.
2. **Transform Jacobian.** Extend `CTransform` with log-Jacobian and gradient; verify with `checkGradients`-style finite differences.
3. **Adaptation and diagnostics.** Dual-averaging step size; acceptance rate; effective sample size and \(\hat R\) as test-side helpers (not in the sampler).
4. **Model enablement.** `CGp` exact first, then `CGplvm`, then sparse and variational models after a check of each objective.
5. **CLI and Python surface.** Sample output as CSV; façade wrapper.
6. **Docs and tutorial.** A short tutorial sampling kernel hyperparameters on the sinc example.

## Backward Compatibility

Purely additive. No existing class changes except `CTransform`, which gains new virtual methods with defaults (Jacobian zero for identity). Existing `.model` files and optimiser paths are untouched.

## Testing Strategy

Verification is against known answers, in keeping with the assert-and-verify tenet.

- **Gradient check.** `checkGradients` on each enabled model at the sampled point, including the Jacobian term.
- **Analytic target.** A test `COptimisable` whose objective is a known Gaussian (and a banana-shaped density): sample mean, variance and covariance must match to Monte Carlo error. Use an `equals`-style tolerance scaled by effective sample size, not `MATCHTOL`.
- **Detailed balance / energy error.** Leapfrog energy error must scale as \(\epsilon^2\); a test shrinks the step size and checks the scaling.
- **Reference comparison.** For a one-dimensional hyperparameter (e.g. RBF length scale on a small data set), compare the HMC marginal against a fine grid evaluation of the posterior computed with the same objective.
- **Invalid-target assertions.** Sampling with no prior on an unbounded parameter must fail loudly.
- **Build gate.** `make` and the new `testHmc` binary must run on this machine under the usual `CHECKS` target.

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md): runnable verification. [REQ-001C](../requirements/req001C_posterior-samples-of-gp-hyperparameters.md) covers sampling.

## Implementation Status

- [x] Phase 0: requirement drafted (REQ-001C)
- [x] Phase 1: core sampler `CHmc` with analytic-target, energy-error and rejection tests (`testHmc`)
- [x] Phase 2: transform Jacobian (`CTransform::logJacobian`, `gradLogJacobian`, `isBounded`) with finite-difference tests
- [x] Phase 3 (partial): dual-averaging step size. ESS is computed in the tests, not in the sampler
- [x] Priors on `likelihood.variance`: `CNoise` is `CRegularisable`, as GPy allows a prior on any parameter
- [x] `CPosteriorTarget`: Jacobian switch (`setIncludeJacobian`, default on; off reproduces GPy) and improper-posterior check (`setAllowImproper` overrides)
- [x] Fix: kernel copies keep their priors (`CRegularisable` deep-copies; `CDist::clone` is now pure virtual)
- [x] Phase 4 (partial): exact FTC `CGp` with kernel parameters and `likelihood.variance`. Other models throw `NotImplementedError`
- [x] Phase 4 (sparse): DTC, DTCVAR, FITC and PITC with fixed inducing points. Sampled parameters are kernel parameters and `likelihood.variance`, reported as 1/beta via `CReciprocalExpTransform`; the noise prior is evaluated at variance = 1/beta. Inducing inputs have no prior, so free inducing points are refused. `gp sample` holds them fixed. Verified by finite-difference gradients, prior-value and sampler checks in `testHmc`
- [x] Phase 4 (GPLVM): plain MAP `CGplvm` samples kernel parameters and latent positions `X.i.j`; the N(0, I) latent regulariser is the prior on X (without it the target is flagged improper). Dynamics, back constraints and learnt input scales refuse. The latent posterior is multimodal (rotation and sign symmetries), so a chain explores one mode. Verified in `testHmc`
- [ ] Phase 4: variational and PPA models
- [x] Priors on noise in the `.model` stream: additive optional `numPriors` key written only when noise priors exist, so prior-free files are byte-identical and old files load unchanged (no version bump needed; old readers cannot load files that carry noise priors)
- [x] Phase 5 (partial): CSV sample output and `gp sample` CLI mode
- [x] Phase 5: Python wrapper `gpc.inference` (`set_prior`, `hmc`, `HmcResult.to_csv`) for `GPRegression` and `SparseGPRegression`; verified against grid integration in `python/tests/test_hmc.py`
- [ ] Follow-up: priors and samples in GPy lengthscale units (today they live on the native inverse width, named `rbf.inv_lengthscale`; a prior on `rbf.lengthscale` is refused). See backlog `2026-11-25_cip003b-lengthscale-prior-units`
- [x] Phase 6: tutorial [hmc_sinc](../docs/tutorials/hmc_sinc.md) (CLI and Python), numbers taken from an actual run

## GPy comparison

GPy attaches priors to any parameter (including `likelihood.variance`) on the constrained value and, as far as we recall, samples in transformed space without a Jacobian and without a properness check. GPc follows the same prior model but adds the Jacobian by default and refuses improper targets.

## References

- R. M. Neal (2011). *MCMC using Hamiltonian dynamics.* Handbook of Markov Chain Monte Carlo.
- M. D. Hoffman and A. Gelman (2014). *The No-U-Turn Sampler.* JMLR. (Source of the dual-averaging step-size scheme.)
- Deferrals this CIP supersedes: [CIP-0015](./cip0015.md), [CIP-0017](./cip0017.md), [CIP-0033](./cip0033.md), [CIP-0035](./cip0035.md).
