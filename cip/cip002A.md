---
author: "Neil Lawrence"
created: "2026-11-19"
id: "002A"
last_updated: "2026-11-20"
status: "Closed"
compressed: false
related_requirements: ["0015", "0002", "000F", "0014"]
related_cips: ["0026", "0027", "0012", "001B", "001C", "001E", "0010"]
tags:
- cip
- svgp
- svi
- sparse
- inducing
- natural-gradients
- adam
- minibatch
title: "SVGP: explicit q(u) with natural gradients, Adam for hyperparameters, minibatches"
---

# CIP-002A: SVGP — explicit \(q(\mathbf{u})\) with natural gradients, Adam for hyperparameters, minibatches

> Prototype evidence: `CSvgp` + `testSvgp` on branch `prototype-svgp` (Phase 1 below).

## Status

- [x] Proposed - Initial idea documented (with Phase 1 prototype)
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [x] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add a stochastic variational sparse GP (Hensman et al. 2013, 2015) as a new model class `CSvgp`, separate from the collapsed `CGp` paths. \(q(\mathbf{u})\) is explicit and **whitened**; it is updated by **natural-gradient** steps written in the conjugate-computation (CVI) form so the same step serves Gaussian and non-Gaussian likelihoods. Kernel hyperparameters, inducing inputs and likelihood parameters use ordinary gradients with **Adam** on seeded minibatches. The Gaussian full-batch case is verified exactly against the existing `CGp` DTCVAR bound.

**Which requirements?** [REQ-0015](../requirements/req0015_stochastic-variational-sparse-gp.md); verification under [REQ-0002](../requirements/req0002_runnable-verification.md); progress events under [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md).

## Motivation

All GPc sparse approximations are collapsed and full batch; all `COptimisable` drivers (CG, SCG, GD, BFGS, LBFGS) assume deterministic objectives and break under minibatch noise (line searches, curvature pairs). Non-Gaussian sparse models are only reachable through PPA EM ([CIP-0026](./cip0026.md)). SVGP is the standard remedy and is what GPy/GPflow users expect. The repo already has the pieces it needs (inducing inputs, kernel cross-covariance and diagonal gradients, Cholesky adjoints, a collapsed bound to verify against).

## Detailed Description

### Model and bound (whitened)

Inducing inputs \(\mathbf{Z}\) (\(M\times q\)), \(\mathbf{K}_{uu}=\mathbf{L}\mathbf{L}^\top\), \(\mathbf{u}=\mathbf{L}\mathbf{v}\), \(q(\mathbf{v}_j)=\mathcal{N}(\tilde{\mathbf{m}}_j,\tilde{\mathbf{S}}_j)\) per output column \(j\). With \(\mathbf{a}_i=\mathbf{L}^{-1}\mathbf{k}_u(\mathbf{x}_i)\):

\[
\mu_{ij}=\mathbf{a}_i^\top\tilde{\mathbf{m}}_j,\qquad
\sigma^2_{ij}=k_{ii}-\|\mathbf{a}_i\|^2+\mathbf{a}_i^\top\tilde{\mathbf{S}}_j\mathbf{a}_i,
\]
\[
\mathcal{L}=\sum_j\Big[\tfrac{N}{|B|}\sum_{i\in B}\mathbb{E}_{\mathcal{N}(f\mid\mu_{ij},\sigma^2_{ij})}[\log p(y_{ij}\mid f)]
-\mathrm{KL}\big(\mathcal{N}(\tilde{\mathbf{m}}_j,\tilde{\mathbf{S}}_j)\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\big)\Big]+\log p(\boldsymbol\theta).
\]

Kernel matrices follow `CGp` exactly (`compute(Z)` + `jitChol`, `compute(Z, X)`, `diagComputeElement`), so `CWhiteKern` behaves identically (zero cross-covariance, present in \(\mathbf{K}_{uu}\) and \(k_{ii}\)).

**Why whitened.** The Phase 1 draft stored \((\mathbf{m},\mathbf{S})\) and inverted \(\mathbf{K}_{uu}\) and \(\mathbf{S}\) explicitly. With \(M=N\), \(\mathbf{Z}=\mathbf{X}\) (20 grid points, RBF lengthscale 0.8, \(\lambda_{\min}(\mathbf{K}_{uu})\approx4\times10^{-16}\)) it returned an "ELBO" of **+34.96 against an exact log marginal of −11.57** — not a bound. Whitened, the same case gives −11.5667 (exact −11.5667; `CGp` stable DTCVAR −11.5674). In whitened coordinates the factorised precision is \(\mathbf{I}+\sum_i(-2g^{v}_i)\mathbf{a}_i\mathbf{a}_i^\top\) with eigenvalues \(\ge 1\) for log-concave likelihoods, and \(\mathbf{K}_{uu}\) is only ever used through triangular solves.

### Natural gradients for \(q(\mathbf{v})\) only

\(q(\mathbf{v})\) is Gaussian, so the natural gradient w.r.t. natural parameters \(\boldsymbol\theta=(\tilde{\mathbf{S}}^{-1}\tilde{\mathbf{m}},\,\tilde{\mathbf{S}}^{-1})\) equals the ordinary gradient w.r.t. expectation parameters. A step of size \(\gamma\in(0,1]\) is

\[
\boldsymbol\theta\leftarrow(1-\gamma)\boldsymbol\theta+\gamma\big(\boldsymbol\theta_{\text{prior}}+\boldsymbol\lambda\big),\quad
\boldsymbol\theta_{\text{prior}}=(\mathbf{0},\mathbf{I}),
\]
\[
\boldsymbol\lambda_{\text{mean}}=c\sum_{i\in B}\mathbf{a}_i\,(g^{\mu}_i-2g^{v}_i\mu_i),\qquad
\boldsymbol\lambda_{\text{prec}}=-2c\sum_{i\in B}g^{v}_i\,\mathbf{a}_i\mathbf{a}_i^\top,\qquad c=N/|B|,
\]

where \(g^\mu_i, g^v_i\) are derivatives of \(\mathbb{E}_q[\log p(y_i\mid f_i)]\) w.r.t. the marginal mean and variance (Khan & Lin 2017). Consequences:

- **Gaussian, full batch, \(\gamma=1\)**: one step lands on the Titsias optimum; ELBO = DTCVAR bound. This is the primary oracle.
- **Minibatch**: the step is linear in the per-point sites, so batch steps are unbiased in natural-parameter space; use \(\gamma<1\) with a decaying schedule.
- **Non-Gaussian**: warm up \(\gamma\) from small values (Salimbeni et al. 2018). Positive definiteness holds for \(\gamma\le1\) when \(g^v_i\le0\), i.e. log-concave likelihoods. All current noise models except `CNcnmNoise` qualify; NCNM is out of scope for the first delivery (would need damping / clipping).
- **Not** for hyperparameters, \(\mathbf{Z}\), or noise parameters: those are not exponential-family variational parameters and use Adam.
- Natural gradients are invariant to the affine whitening map, so whitening changes conditioning, not the step.

**Naming.** `CBayesGplvm` uses "natural grads" for gradients in untransformed ("natural") parameter space (`computeVarDtcGrads`). Rename that usage (e.g. "untransformed grads") before this CIP lands to avoid confusion with Fisher-preconditioned natural gradients.

### Hyperparameter and inducing-input gradients

Back-propagate \(\partial\mathcal{L}/\partial\mu_{ij}, \partial\mathcal{L}/\partial\sigma^2_{ij}\) to \(\partial\mathcal{L}/\partial\mathbf{K}_{uf}\), \(\partial\mathcal{L}/\partial k_{ii}\), and through \(\mathbf{L}^{-1}\) to \(\partial\mathcal{L}/\partial\mathbf{K}_{uu}\) using the adjoints from [CIP-0012](./cip0012.md): `trsmAdjointUT` is exactly the adjoint of the whitened projection \(\mathbf{A}=\mathbf{U}^{-\top}\mathbf{K}_{uf}\) (giving \(\bar{\mathbf{K}}_{uf}\), \(\bar{\mathbf{U}}\)), and `cholAdjointUpperBlocked` maps \(\bar{\mathbf{U}}\) to \(\bar{\mathbf{K}}_{uu}\) (mind its tril-pairing convention: halve off-diagonals before kernel contractions). Kernel parameters then use `getGradTransParams` (cross and symmetric) and `getDiagGradTransParams`; \(\mathbf{Z}\) uses `getGradX`. Verified by finite differences on a small problem (full batch, fixed \(q\)).

### Stochastic optimiser

`COptimisable` is not reused: its contract is a deterministic full objective. Add a small stochastic interface — `numData()`, `objectiveGradBatch(batch, g)`, `getOptParams/setOptParams` — and an `CAdam` driver with: seeded `mt19937_64` sampler (epoch-wise permutation, no replacement), step schedules, optional interleaved natural-gradient step on \(q\) each iteration, and outer-iteration progress events through the [CIP-001E](./cip001E.md) observer. No optimiser logic inside objective/gradient evaluation.

### Likelihoods

`CNoise::logLikelihood(mu, var, y)` returns \(\log\mathbb{E}_q[p(y\mid f)]\) (ADF/IVM/PPA quantity). SVGP needs \(\mathbb{E}_q[\log p(y\mid f)]\). Add a separate virtual, e.g. `expectedLogLikelihood(y, mu, var, gMu, gVar)`, closed form for Gaussian and Gauss–Hermite (default 20 nodes, configurable) for probit and ordered categorical. Keep the two quantities distinctly named.

Gaussian noise is the precision \(\beta\) today; align with [CIP-0027](./cip0027.md) (`Gaussian_noise.variance`) when that lands. Prefer no `CWhiteKern` in SVGP defaults; tests cover both.

### Relation to sparse PPA

Sparse PPA's E-step solves a Gaussian-site regression for \(q(\mathbf{u})\) in closed form: a full-batch \(\gamma=1\) natural-gradient step on the conjugate part. SVGP-probit and sparse PPA can therefore be compared on identical problems; the exact probit marginal (Gaussian orthant probability, used to check the dense PPA bound on 2026-11-18) bounds both from above.

### Prototype evidence (Phase 1, `testSvgp`)

| Check | Result |
|---|---|
| \(\gamma=1\) full batch vs `CGp` DTCVAR (\(N=60\), \(M=12\), \(d=2\)), rbf+bias / rbf+bias+white | diff \(9\times10^{-9}\) / \(3\times10^{-11}\) |
| Titsias optimum is a fixed point | \(<10^{-9}\) |
| Predictive mean / latent variance vs `CGp` (after removing CGp's added \(1/\beta\)) | \(\le 3\times10^{-14}\) |
| \(\gamma=0.3\): monotone ELBO, converges to bound in 60 steps | yes |
| Minibatch (5-way partition): ELBO and natural-param steps average to full batch | rel. \(\le 4\times10^{-14}\) |
| \(M=N\), \(\mathbf{Z}=\mathbf{X}\), well conditioned: ELBO = exact log marginal | −18.0652 = −18.0652 |
| \(M=N\), \(\mathbf{K}_{uu}\) numerically singular | −11.5667 vs exact −11.5667 (draft: +34.96) |

Note: `CGp::posteriorMeanVar` returns variance **including** observation noise \(1/\beta\); `CSvgp::posteriorMeanVar` returns latent variance. Tracked in backlog [`2026-11-19_latent-vs-predictive-meanvar`](../backlog/features/2026-11-19_latent-vs-predictive-meanvar.md): split into `latentMeanVar` / `predictiveMeanVar` (GPy `predict_noiseless` / `predict`).

### Tenets

- **Assert and verify** — exact DTCVAR oracle, exact log marginal at \(M=N\), finite-difference gradients, GPy `SVGP` fixture.
- **Compose through interfaces** — reuse `CKern`, inducing inputs, `CCholAdjoint`; new stochastic optimiser interface rather than bending `COptimisable`.
- **Flexibility without a performance tax** — per-step cost \(O(|B|M^2+M^3)\); no \(N\times N\) storage; no allocation in inner loops beyond \(M\times M\) work buffers.
- **LAPACK owns linear algebra** — triangular solves and Cholesky only; no explicit \(\mathbf{K}_{uu}^{-1}\).
- **Ecosystem** — GPy `SVGP` parity; parameter paths per CIP-0014.

## Implementation Plan

1. **Phase 1 (prototype, done on `prototype-svgp`)** — `CSvgp`: whitened \(q\), ELBO (full / minibatch), CVI natural-gradient step, predictions, `setQ`/`getQ`; Gaussian only; `testSvgp`.
2. **Phase 2 — gradients** — ELBO gradients for kernel parameters, \(\mathbf{Z}\), \(\beta\) (via `CCholAdjoint`); `checkGradients`-style finite-difference test; full-batch Adam + natgrad recovers DTCVAR hyperparameter optimum on a small problem.
3. **Phase 3 — stochastic loop** — stochastic optimiser interface, `CAdam`, seeded sampler, schedules, observer events; reproducibility test; convergence smoke on \(N=10^4\).
4. **Phase 4 — non-Gaussian** — `expectedLogLikelihood` with Gauss–Hermite for probit and ordered; \(\gamma\) warm-up; compare against exact probit marginal and sparse PPA on shared problems; GPy `SVGP` fixture.
5. **Phase 5 — surfaces** — CLI (`gp learn --approx svgp -a M --batch B --seed S` or a dedicated binary), stream version, Python façade (CIP-001D), docs/tutorial.

## Backward Compatibility

- New class and files only; `CGp`, `CPpa`, `CSparsePpa`, `COptimisable` behaviour unchanged.
- The `CBayesGplvm` "natural grads" rename is internal (comments / local names); no stream or CLI change.

## Testing Strategy

- `testSvgp` (Phase 1, matio-free): DTCVAR oracle, fixed point, predictions, monotone \(\gamma<1\), minibatch unbiasedness, \(M=N\) exact and ill-conditioned cases, guards. Add to `matio-tests` / coverage when accepted.
- Phase 2: finite-difference gradients (kernel, \(\mathbf{Z}\), \(\beta\)) at fixed \(q\).
- Phase 3: same seed ⇒ identical trajectory; different seeds ⇒ same optimum within tolerance on a Gaussian problem.
- Phase 4: probit ELBO \(\le\) exact orthant log marginal on small problems; GPy `SVGP` fixture parity.

## Related Requirements

- [REQ-0015](../requirements/req0015_stochastic-variational-sparse-gp.md) — owner.
- [REQ-0002](../requirements/req0002_runnable-verification.md) — runnable verification.
- [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md) — progress events for the stochastic loop.
- [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md) — noise in the likelihood.

## Implementation Status

- [x] Phase 1 prototype: `CSvgp` + `testSvgp` (branch `prototype-svgp`)
- [x] CIP accepted
- [x] Phase 2: hyperparameter / Z / β gradients (FD max rel ~7e-9)
- [x] Phase 3: CAdam + seeded minibatches + progress observer
- [x] Phase 4: expectedLogLikelihood + Gauss–Hermite probit
- [x] Phase 5: tutorial `docs/tutorials/svgp.md` (CLI/stream/Python façade deferred to follow-up)
### 2026-11-19 (implementation)
Phases 2–4 landed on `cip002A-svgp` with `testSvgp` green (FD grads, Adam seed parity, probit GH). Tutorial under `docs/tutorials/svgp.md`. CLI/stream/Python façade remain as a light follow-up under the Phase 5 backlog task.

### Follow-on (Proposed)
- Route `CSvgp` \(K_{uu}/K_{uf}\) through `CInducing` so white-noise / Green’s / weight-space inducing work under SVGP — backlog [`2026-11-19_cip002A-inducing-interface`](../backlog/features/2026-11-19_cip002A-inducing-interface.md) (from CIP-0028 review).
