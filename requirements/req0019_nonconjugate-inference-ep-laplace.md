---
id: "0019"
title: "Non-conjugate inference on CGp: EP and Laplace over pure likelihoods"
status: "Proposed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces", "assert-and-verify"]
stakeholders: ["maintainers", "modellers", "gpy-porters", "teachers"]
tags: ["ep", "laplace", "likelihood", "inference", "classification", "probit", "architecture", "gpy", "gpy2"]
---

# REQ-0019: Non-conjugate inference on CGp: EP and Laplace over pure likelihoods

## Description

GPc provides **Expectation Propagation (EP)** and **Laplace** approximation for non-conjugate likelihoods on the dense / sparse `CGp` path, with a soft-classification story comparable to GPy `GPClassification` (Bernoulli / probit + EP or Laplace + RBF, no trained White softener required).

Likelihood objects are **parameters plus batched pure functions** (log density, variational expectations, EP moments, predictive moments). Per-point sites and ADF caches live in **inference workspaces**, not on the likelihood. Targets and masks have **one data owner** per model ([docs/target-architecture.md](../docs/target-architecture.md) rules 1–4).

**Why this matters**: CIP-0027 / [REQ-0014](./req0014_observation-noise-in-likelihood.md) put Gaussian observation noise on the likelihood but explicitly deferred EP/Laplace and stripping sites out of `CNoise`. Classification tutorials still point at IVM + White-in-\(K\) as a substitute. Ecosystem interoperability and the shared GPc / GPy2 architecture both require non-conjugate inference built on likelihood primitives, not stateful `CNoise` copies of \(Y\), \(\mu\), \(\mathrm{var}\Sigma\).

**Who benefits**: Modellers porting GPy classification; teachers who want EP/Laplace demos without IVM; maintainers aligning IVM / PPA / SVGP on one likelihood API; a future GPy2 that shares fixtures with GPc.

## Acceptance Criteria

- [ ] `CNoise` (or successor likelihood type) holds **parameters only** on the hot path: no owned `py`, no per-point `mu` / `varSigma` site caches as likelihood state
- [ ] Batched likelihood API includes at least: `log_density`, `variational_expectations` (already sketched), **`ep_moments`**, and predictive moments usable by EP / Laplace / `out`
- [ ] EP inference on `CGp` for probit (or Bernoulli–probit link) trains kernel hyperparameters with a verified bound / marginal approximation; default kernel need not include White
- [ ] Laplace approximation on `CGp` for the same likelihood class, with mode-finding and Hessian (or equivalent) verified against FD or known fixtures
- [ ] Dense FTC classification smoke matches a GPy-generated or orthant / known-good fixture within documented tolerance
- [ ] Softness story is explicit: White-in-\(K\) (IVM exception) vs likelihood scale vs site precision — documented, not mixed silently
- [ ] Old streams / IVM / PPA keep working during migration (adapters or dual-read); no silent behaviour flip for existing IVM White defaults
- [ ] Non-goals for first delivery: full GPy likelihood zoo (Poisson, Student-t, warped, heteroscedastic); HMC; replacing SVGP / PPA / IVM; Python façade beyond a thin classify/predict bind; putting EP inside the Python hot loop

## Notes (Optional)

HOW: [CIP-0033](../cip/cip0033.md). Builds on [CIP-0027](../cip/cip0027.md) Phase 0 (`ndlnoise` helpers), [CIP-002A](../cip/cip002A.md) `expectedLogLikelihood` / Gauss–Hermite, [CIP-0031](../cip/cip0031.md) latent vs predictive. Orthogonal to [CIP-0016](../cip/cip0016.md) file interchange.

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces, assert-and-verify
- **Architecture**: [docs/target-architecture.md](../docs/target-architecture.md)
- **External**: GPy `GPClassification`, EP / Laplace; Rasmussen & Williams ch. 3; Minka EP

## Progress Updates

### 2026-11-21

Requirement drafted with CIP-0033 (spawned from CIP-0027 deferred follow-on).
