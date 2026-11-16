---
id: "000C"
title: "Users can train Manifold Relevance Determination (GPy-parity subset)"
status: "In Progress"
priority: "Medium"
created: "2026-11-16"
last_updated: "2026-11-16"
related_tenets: ["compose-through-interfaces", "assert-and-verify", "ecosystem-interoperability", "lapack-owns-linear-algebra"]
stakeholders: ["maintainers", "modellers"]
tags: ["gplvm", "bayesian", "mrd", "multi-view", "ard", "variational", "gpy"]
---

# REQ-000C: Users can train Manifold Relevance Determination (GPy-parity subset)

## Description

GPc provides a multi-view variational latent-variable path comparable to GPy’s `MRD`: one shared diagonal Gaussian \(q(\mathbf{X})\) and shared inducing inputs \(Z\) across \(V \ge 2\) aligned observation matrices \(\{\mathbf{Y}^{(v)}\}\), each with its own kernel (default RBF-ARD) and homoscedastic Gaussian noise. Soft private versus shared latent structure emerges by comparing per-view ARD input scales after training.

This builds on the single-view Bayesian GP-LVM delivered by [REQ-000B](./req000B_bayesian-gplvm-psi.md) / [CIP-0017](../cip/cip0017.md). MRD is not new psi maths; it is multi-view composition with a single KL term.

**Why this matters**: Interoperate With the GP Ecosystem — GPy tutorials and papers treat MRD as the natural multi-view extension of BayesianGPLVM. Compose Through Interfaces — reuse the existing psi / VarDTC path rather than forking a second inducing solver. Assert and Verify — fixture compare against GPy `MRD` on a tiny two-view problem.

**Who benefits**: Modellers with aligned multi-modal or multi-sensor data who need soft shared/private latent factorisation without hard subspace constraints.

## Acceptance Criteria

- [x] Callers can supply \(V \ge 2\) observation matrices with equal \(N\) (feature dims \(D_v\) may differ) and train a shared \(q(\mathbf{X})\) and \(Z\)
- [x] Each view has its own kernel (at least RBF-ARD) and noise precision; objective is \(\sum_v \mathcal{L}_{\mathrm{VarDTC}}^{(v)} - \mathrm{KL}(q\|p)\) with the KL counted once
- [x] Gradients for shared \(Z,\boldsymbol{\mu},\mathbf{S}\) are sums of per-view contributions; per-view \(\theta_v,\beta_v\) stay local
- [x] Init supports concat-PCA (and documented alternative) of the views into shared \(\boldsymbol{\mu}\); \(Z\) from subset/permute of \(\boldsymbol{\mu}\)
- [x] Callers can read per-view ARD scales and classify latent dims as shared / private given a threshold (GPy `factorize_space` spirit)
- [ ] Verification artefacts compare ELBO and/or ARD scale ranking to a GPy-generated text fixture on a seeded two-view toy, without MATLAB at runtime
- [x] Non-goals are explicit for the first delivery: missing-data VarDTC, non-Gaussian views, minibatch/SVI/MPI, Linear-ARD psi, SSGPLVM, dynamics priors, Matérn/tensor psi, Python bindings

## Notes (Optional)

HOW is [CIP-0018](../cip/cip0018.md). Assessment of distance from CIP-0017: multi-view wiring is the blocker; RBF-ARD psi already exists. Prefer an explicit single-KL joint objective over GPy’s `kl_factr = 1/V` child hack.

## References

- **Related Tenets**: compose-through-interfaces, assert-and-verify, ecosystem-interoperability, lapack-owns-linear-algebra
- **External Links**: Damianou et al. (2012), *Manifold Relevance Determination*, ICML; GPy `models.mrd`, `models.bayesian_gplvm`

## Progress Updates

### 2026-11-16
Requirement created from the CIP-0017 vs GPy / MRD assessment. Status Proposed; implementation owned by CIP-0018.

### 2026-11-16 (implementation)
`CMrd` + `testMrd` / `demo_mrd` on branch `cip0018-mrd`. Fixture generator added; committed GPy ELBO compare still open.
