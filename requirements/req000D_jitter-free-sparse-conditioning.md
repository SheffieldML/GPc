---
id: "000D"
title: "Sparse GPs can use inducing variables whose conditioning does not depend on kernel hyperparameters"
status: "Proposed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets: ["compose-through-interfaces", "assert-and-verify", "lapack-owns-linear-algebra", "flexibility-with-performance", "ecosystem-interoperability"]
stakeholders: ["maintainers", "modellers"]
tags: ["sparse", "inducing", "inter-domain", "numerics", "jitter", "dtcvar", "fitc"]
---

# REQ-000D: Sparse GPs can use inducing variables whose conditioning does not depend on kernel hyperparameters

## Description

With point inducing variables, \(K_{uu}=k(Z,Z)\) becomes ill-conditioned as lengthscales grow relative to inducing spacing, as ARD switches dimensions off, or as inducing points collapse. GPc currently papers over this with jitter (`_updateInvK`, the "jitter of … added to K_uu" warnings) and with the whitened StableFactor / LQ paths ([CIP-0012](../cip/cip0012.md), [CIP-001B](../cip/cip001B.md)), which stabilise everything *downstream* of \(K_{uu}\) but still factor \(K_{uu}\) itself.

Jitter is equivalent to redefining the inducing variables as noisy observations, and the collapsed bound is monotonically decreasing in that noise (Burt, Rasmussen & van der Wilk 2020, Prop. 34), so jitter is a pure loss that should be avoided rather than tuned.

Users should be able to choose an inducing-variable family for which \(K_{uu}\) depends only on the inducing parameters (not on lengthscale or signal variance), so hyperparameter learning cannot drive \(K_{uu}\) towards singularity. Point inducing variables remain the default and are one member of the same contract.

This requirement addresses **kernel-driven** ill-conditioning of \(K_{uu}\). Collapse of inducing centres (or badly chosen window widths) can still singularise a kernel-free \(K_{uu}\); that remains a complementary problem (e.g. minimum-separation initialisers), not a substitute for this requirement.

**Who benefits**: modellers whose sparse fits fail mid-optimisation with Cholesky/jitter warnings as lengthscales grow; maintainers who want the stability suite's crowded-\(Z\) / high-\(\gamma\) cells to have a structural fix, not only a numerical one.

## Acceptance Criteria

- [ ] Inducing variables are a contract consumed by `CGp` (point inducing is one implementation); DTC, DTCVAR, FITC and PITC run unchanged on any implementation
- [ ] At least one non-point family is available for RBF / RBF-ARD with \(K_{uu}\) independent of kernel hyperparameters
- [ ] A reproducibility check shows the new family reduces exactly to point inducing at a documented parameter setting (log-likelihood and total-derivative gradients agree to `ndlutil::MATCHTOL`-level tolerance)
- [ ] On a lengthscale sweep with fixed inducing parameters, the new family needs no jitter on \(K_{uu}\) where point inducing does (numerical-stability suite cell); ELBO / bound quality vs (jittered) point inducing is recorded on the same cell
- [ ] An effective-inducing-count diagnostic is exposed so users can see wasted capacity
- [ ] Save/load round-trips the inducing family and its parameters; existing `.model` files load unchanged
- [ ] Non-goals for first delivery are explicit (weight-space / network features, causal Green's-function features, psi statistics with new features)

## Notes (Optional)

HOW is [CIP-001C](../cip/cip001C.md). Follow-on CIPs (Green's-function features; weight-space features for dot-product kernels) should also cite this requirement.

## References

- **Related Tenets**: compose-through-interfaces, assert-and-verify, lapack-owns-linear-algebra, flexibility-with-performance, ecosystem-interoperability
- **External Links**: Burt, Rasmussen & van der Wilk (2020) JMLR; Lázaro-Gredilla & Figueiras-Vidal (2009); van der Wilk et al. (2020) *A framework for interdomain and multioutput GPs*

## Progress Updates

### 2026-11-18
Requirement created alongside CIP-001C (Proposed). Pilot worktree `cip001C-wn-pilot` opened for a lengthscale-sweep spike before acceptance.
