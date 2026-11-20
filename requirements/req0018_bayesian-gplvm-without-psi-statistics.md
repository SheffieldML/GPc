---
id: "0018"
title: "Bayesian GP-LVM can be trained without closed-form psi statistics"
status: "In Progress"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
related_tenets: ["assert-and-verify", "compose-through-interfaces", "flexibility-with-performance", "lapack-owns-linear-algebra"]
stakeholders: ["maintainers", "modellers", "GPy users", "teachers"]
tags: ["bayesian-gplvm", "svi", "uncollapsed", "reparameterisation", "inducing", "weight-space", "minibatch"]
---

# REQ-0018: Bayesian GP-LVM can be trained without closed-form psi statistics

## Description

Users can fit a Bayesian GP-LVM, with a Gaussian \(q(\mathbf X)\) over latent points, using any kernel and any inducing family GPc supports. That means the kernels and families that have **no closed-form psi statistics**: the MLP kernel with weight-space ridge features ([CIP-0028](../cip/cip0028.md)), white-noise windows ([CIP-001C](../cip/cip001C.md)), Green's-function features ([CIP-0029](../cip/cip0029.md)), and point inducing with kernels outside the CIP-0017 psi set.

The bound is the uncollapsed SVGP evidence lower bound with latent inputs integrated by sampling, so it can also be optimised on minibatches.

**Why this matters.** Today `CBayesGplvm` / `CMrd` call `CKern::psi0/1/2` with point inducing \(\mathbf Z\). They work only for White, Bias, Linear, RBF, RBF-ARD and compounds of these, and never with `CInducing` families. The MLP kernel's psi statistics are not available in closed form, not even \(\psi_0\). Weight-space MLP features, the main motivation for CIP-0028, therefore cannot be used for latent-variable models at all.

**Who benefits.** Modellers wanting MLP or other non-RBF latent-variable models; large-\(N\) Bayesian GP-LVM users (minibatches); teachers comparing collapsed and uncollapsed bounds.

## Acceptance Criteria

- [ ] A Bayesian GP-LVM variant optimises the uncollapsed ELBO with explicit \(q(\mathbf u)\), Gaussian \(q(\mathbf X)\) and reparameterised samples of \(\mathbf X\), for every `CInducing` family and every kernel with `getDiagGradX`
- [ ] Where closed-form psi statistics exist (RBF / RBF-ARD, point inducing, Gaussian noise), the estimator's expectation equals the exact uncollapsed ELBO computed from \(\Psi_0,\Psi_1,\Psi_2\); at the collapsed optimum of \(q(\mathbf u)\) it equals `CBayesGplvm`'s bound
- [ ] In the limit \(S\to0\) it reduces to SVGP at inputs \(\boldsymbol\mu\) (and to `CGp` DTCVAR at the \(q(\mathbf u)\) optimum)
- [ ] Gradients for \(\boldsymbol\mu\), \(S\), kernel, inducing and likelihood parameters pass finite-difference checks at fixed random numbers, for each inducing family
- [ ] Minibatches over data points and a seeded sampler give unbiased, reproducible estimates
- [ ] Gaussian and probit likelihoods are supported (probit via the existing Gauss–Hermite path)
- [ ] Non-goals are explicit: amortised or back-constrained \(q(\mathbf X)\), dynamics priors, deep GPs, GPU backends ([REQ-0016](./req0016_gpu-training-decision.md)), SSGPLVM, changes to collapsed `CBayesGplvm` behaviour or defaults

## Notes (Optional)

HOW is [CIP-0030](../cip/cip0030.md). [REQ-000B](./req000B_bayesian-gplvm-psi.md) lists SVI as a non-goal of the collapsed delivery; this requirement owns it. It builds on [REQ-0015](./req0015_stochastic-variational-sparse-gp.md) (SVGP). Deep GPs remain a non-goal; they compose the same Phase 0b SVGP core later.

## References

- **Related Tenets**: assert-and-verify, compose-through-interfaces, flexibility-with-performance, lapack-owns-linear-algebra
- **External Links**: Titsias & Lawrence (2010), *Bayesian Gaussian Process Latent Variable Model*; Hensman, Fusi & Lawrence (2013), *Gaussian Processes for Big Data*; Titsias & Lázaro-Gredilla (2014), *Doubly Stochastic Variational Bayes for non-Conjugate Inference*; Kingma & Welling (2014), *Auto-Encoding Variational Bayes*

## Progress Updates

### 2026-11-20
Requirement drafted as Proposed alongside CIP-0030.

### 2026-11-20 (later)
Moved to `requirements/`; status In Progress; priority High (unlocks CIP-0028 latent-variable use). CIP-0030 Accepted → In Progress.
