# Target architecture: GPc and a JAX-based GPy2

One design, two engines: a JAX-based GPy2 for GPU and research work, with GPc as the verified CPU reference and embeddable engine.

**Status:** living note (compressed 2026-11-21 for CIP-0027 / 0033 / 0034). CIPs that change model structure, likelihoods, data handling, randomness or serialisation should cite this note and say how they conform or why they deviate.
**Relates to:** [CIP-0027](../cip/cip0027.md) (noise in the likelihood) · [CIP-0033](../cip/cip0033.md) / [CIP-0034](../cip/cip0034.md) / [REQ-0019](../requirements/req0019_nonconjugate-inference-ep-laplace.md) (likelihood/inference split; EP/Laplace; site strip) · [CIP-002D](../cip/cip002D.md) Part B and [REQ-0016](../requirements/req0016_gpu-training-decision.md) (GPU route) · [CIP-002A](../cip/cip002A.md) (SVGP) · [CIP-0014](../cip/cip0014.md) (parameter paths) · [CIP-0009](../cip/cip0009.md) (caches) · [CIP-0030](../cip/cip0030.md) (uncollapsed Bayesian GP-LVM).

## Why this note exists

GPc is aligning with GPy. GPy1's structure (stateful parameters, data stored on models, inference inside model classes) is not where a GPU-capable successor would go. A JAX-based "GPy2" would follow the functional pattern of current JAX GP libraries:

- **GPJax:** the posterior is the product of a prior and a likelihood; data lives in a `Dataset` and enters at the objective. See its [regression example](https://jaxgaussianprocesses.github.io/GPJax/examples/regression).
- **tinygp:** deliberately ships no inference routines, only composable kernels and operations. See [Why tinygp?](https://tinygp.readthedocs.io/en/v0.1.0/motivation.html).

This note fixes the target GPc should converge on, so that GPc and GPy2 share one decomposition, one parameter tree, one set of objective signatures, and one set of test fixtures. Under CIP-002D Part B that is option B: GPU work happens in GPy2, and GPc is the reference.

## The decomposition

| Part | Role | Holds |
|---|---|---|
| **Dataset** | inputs, targets, missing-data mask, normaliser | data only; passed to objectives in batches |
| **Kernel, mean function** | the prior | parameters only |
| **Inducing features** | \(K_{uu}\), \(K_{uf}\), dispatched on (feature type, kernel type) | feature parameters only (e.g. centres, directions, window width) |
| **Likelihood** | pure, batched functions of (parameters, \(f\) or \((\mu,v)\), \(y\)) | parameters only (e.g. Gaussian variance, probit bias) |
| **Variational families** | \(q(\mathbf u)\) (whitened), \(q(\mathbf X)\) | variational parameters only |
| **Covariance operator + solver** | dense, low-rank plus diagonal, …; Cholesky, LQ, whitened | nothing persistent; custom gradients where stability needs them |
| **Objective** | `objective(params, batch, key) -> value` (+ gradients) | nothing; exact, collapsed (Titsias), SVGP and Bayesian GP-LVM bounds are separate functions |
| **Inference state** | EP / ADF sites, PPA \(\langle f\rangle\) | its own workspaces, separate from the likelihood |
| **Optimiser** | gradient steps (Adam / optax), natural-gradient steps on \(q(\mathbf u)\) | optimiser state |

The parameters of every part form **one parameter tree** with constraint transforms. GPy1-style mutable access (`model.Gaussian_noise.variance = x`, `.fix()`) is sugar in the Python façade, not the core model.

## Rules

1. **One data owner.** Targets, mask and normalisation live in exactly one place per model, computed once. No likelihood, inference object or helper keeps a copy of per-point data. No new phase may add one.
2. **Likelihoods are parameters plus batched functions.** The working interface takes contiguous blocks \((Y_B,\mu_B,v_B)\) and writes values and derivatives into caller-provided buffers: log density and derivatives, \(\mathbb E_q[\log p]\) (`variational_expectations`), \(\log\mathbb E_q[p]\) (`log_predictive_density`), EP moments, predictive moments. Per-element methods remain only as test oracles. Likelihoods hold no data and no mutable state.
3. **Contiguous blocks.** Callers gather a minibatch once and pass blocks, not index lists into shared storage.
4. **Inference state belongs to inference.** Site parameters and similar per-point state live in the inference code's workspaces ([CIP-002D](../cip/cip002D.md) QW7), not in the likelihood.
5. **Pure objectives.** The same parameters and batch give the same value and gradients regardless of call history. Caches ([CIP-0009](../cip/cip0009.md)) are allowed only as invisible optimisations. A purity test (evaluate, perturb, restore, re-evaluate) belongs in each model's suite.
6. **Explicit randomness.** Stochastic objectives take their randomness per call (a seed or key) and accept injected noise \(\boldsymbol\epsilon\), so GPc and GPy2 can evaluate the same estimator on the same noise and match to round-off. No RNG hidden inside model objects as the only source.
7. **Canonical parameter tree, GPy1 names as aliases.** Paths are structural (e.g. `likelihood.variance`); GPy1 spellings (`Gaussian_noise.variance`) map onto them ([CIP-0014](../cip/cip0014.md)). Each quantity has one canonical parameterisation (Gaussian noise: variance), with derived forms (precision \(\beta\)) computed internally where the algebra wants them.
8. **Gradients cross-check.** GPc's hand-derived adjoints and GPy2's autodiff test each other. Shared fixtures record values and gradients at fixed parameters, data and \(\boldsymbol\epsilon\).
9. **Serialisation separates parameters from data.** Model files hold configuration plus the parameter tree; data is a separate, optional block. Readers keep accepting older formats.
10. **Precision is explicit.** FP64 by default (GPy2 enables 64-bit floats). Any mixed-precision path keeps factorisations, log-determinants and triangular solves in FP64 ([CIP-002D](../cip/cip002D.md) B4).

## GPc today → change

| Area | After CIP-0027 / 0033 / 0034 | Still open |
|---|---|---|
| Data | `CNoise::py` is a **non-owning** pointer; models own \(Y\). Scaled targets `CGp::m` etc. remain model-side | Fully unify on a single Dataset object / drop residual Y copies across GPLVM paths |
| Likelihood | Gaussian observation noise on `likelihood.variance` ([CIP-0027](../cip/cip0027.md)); Bernoulli–probit `ep_moments` / Laplace Hessian oracles ([CIP-0033](../cip/cip0033.md)); site matrices **not** likelihood parameters | Broader likelihood zoo; remove pointer field `py` entirely from the type if desired |
| Sites | ADF `mu`/`varSigma` live in `CNoiseSiteWorkspace` — owned fallback or attached by IVM / PPA / SparsePpa ([CIP-0034](../cip/cip0034.md)) | — |
| Noise parameter | FTC Gaussian: `likelihood.variance`; sparse \(\beta\) derived. PPA augmentation \(\beta\) stays | — |
| Classification | Dense FTC EP / Laplace on `CGp`; `gpc.GPClassification` | Sparse EP / EPDTC deferred |
| Objective / RNG | Cache protocol; SVGP still has internal RNG paths | Rule 6 purity / injectable \(\boldsymbol\epsilon\) everywhere |
| Gradients | Hand-written adjoints kept | Cross-check fixtures vs GPy2 autodiff |
| Serialisation | `CNoise` streams **parameters only** (not sites / Y) | Broader model file split (config + params vs data) |

## What not to copy from GPy1

- Stateful parameter objects with observers and in-place updates (keep as façade sugar only).
- Data stored on models (`model.X`, `set_XY`).
- Inference baked into model classes (`GPRegression` = model + exact inference).
- Per-point numerics with `Y_metadata` side channels.

Kept from GPy1: noise belongs to the likelihood; GPy1 names survive as aliases.

## How current and planned CIPs map

| CIP | What it takes from this note |
|---|---|
| **[CIP-0027](../cip/cip0027.md)** (**Closed**, compressed) | Likelihood-owned Gaussian noise; drop auto White/Bias for regression; IVM White softener exception documented. |
| **[CIP-0033](../cip/cip0033.md)** / [REQ-0019](../requirements/req0019_nonconjugate-inference-ep-laplace.md) (**Closed** / **Validated**) | `ep_moments`; FTC EP + Laplace on `CGp`; `gpc.GPClassification`. |
| **[CIP-0034](../cip/cip0034.md)** (**Closed**, compressed) | Site caches in `CNoiseSiteWorkspace`; IVM/PPA attach; closes 0027’s deferred strip. |
| **CIP-0030** | Rule 6: seed per step and injectable \(\boldsymbol\epsilon\) for the sampled Bayesian GP-LVM bound. |
| **CIP-002D Part B / REQ-0016** | Option B is GPy2. Rules 5 and 8 (purity tests, shared fixtures) are the contract between the two engines. |

## Reviewer checklist

For any change touching models, likelihoods, inference, data or serialisation:

- [ ] Does it add per-point storage outside the data owner or inference workspaces? (rule 1, 4)
- [ ] Does it add a per-element virtual call on a hot path? (rule 2)
- [ ] Does it add mutable state to a likelihood, kernel or inducing object beyond parameters? (rules 2, 5)
- [ ] Is every stochastic objective given its randomness explicitly, with \(\boldsymbol\epsilon\) injectable? (rule 6)
- [ ] Are new parameters on the canonical tree with one parameterisation, and GPy1 aliases where they exist? (rule 7)
- [ ] Are values and gradients recorded as shared fixtures where a GPy2 counterpart is planned? (rule 8)
- [ ] Do saved models keep parameters and data separate, with readers for old formats? (rule 9)
