# IVM softener: White-in-\(K\) vs probit scale vs EP

Supplementary notes for [CIP-0027](../cip0027.md). Core decisions stay in the CIP; this file holds the modelling write-up for Phase B′ and for a future non-linear-likelihoods CIP.

## What IVM does today

| Layer | Behaviour |
|-------|-----------|
| Likelihood | `CProbitNoise`: \(p(y\mid f)=\Phi\!\big(y(f+b)/\sqrt{v+\sigma^2_{\mathrm{probit}}}\big)\) with **fixed** \(\sigma^2_{\mathrm{probit}}=10^{-6}\). Trainable noise params: **bias only**. |
| Outer loop | `selectPoints` → optimise kernel (`kernIters`) → optimise noise (`noiseIters` on `pnoise`). |
| Kernel defaults | CLI always appends `CBiasKern` + `CWhiteKern`. |
| Active-set Gram | `Kstore` is filled with `computeElement`. **`CWhiteKern::computeElement` is always 0** (even on the diagonal), so `CIvm::updateM` **patches** the active diagonal with `pkern->getWhite()`. |

So the **trained** softener is White in the prior, not a free probit variance. IVM “likes” White because that is how softness enters both the kernel M-step and the active-set algebra.

## Three different softeners (not interchangeable)

| Mechanism | Story | Effect |
|-----------|--------|--------|
| Unit probit + **White in \(K\)** | \(f\sim\mathcal{N}(0,K+\sigma_w^2 I)\), \(p(y\mid f)=\Phi(yf)\) | Independent prior variance per point; changes posterior correlations. **Current IVM path.** |
| **Scaled probit** (trainable \(\sigma_\ell^2\)) | \(f\sim\mathcal{N}(0,K)\), \(p(y\mid f)=\Phi(yf/\sigma_\ell)\) | Likelihood temperature only. |
| Heaviside + latent Gaussian noise | \(y=\mathrm{sign}(f+\varepsilon)\), \(\varepsilon\sim\mathcal{N}(0,\sigma^2)\) | ≡ scaled probit with that \(\sigma\). **Not** ≡ White-in-\(K\) when \(K\) is correlated (RBF, etc.). |

You cannot rename `white.variance` to `probit.variance` and expect the same IVM behaviour.

## Options (scoped)

### A — Retain trained White for IVM (CIP-0027 preferred outcome)

Regression / PPA / Python FTC drop auto White and Bias. IVM **keeps** trained White (Bias may still drop if B′ says bias is redundant with probit bias). Document explicitly: classification softener ≠ Gaussian measurement noise.

### B — Train probit \(\sigma^2\) inside the existing IVM noise M-step

Add \(\sigma^2_{\mathrm{probit}}\) to `CProbitNoise`’s optimisable vector (positive transform). Site updates already depend on \(\sqrt{v+\sigma^2}\); `noiseIters` would train it without a new EP engine. That is hyperparameter training under the **current site approximation**, not full-dataset EP.

Caveats: landscape partly redundant with kernel variance and with White if both free; **still not equivalent** to White-in-\(K\). B′ may compare this to status quo; shipping it as the default softener is **out of scope for CIP-0027**.

### C — GPy-style Bernoulli + EP / Laplace on `CGp` (deferred)

Default kernel can be RBF-only; softness lives in the likelihood + inference method. That is the real “integration of non-linear likelihoods” project (sites, EP/Laplace/variational, gradients, sparse + non-conjugate). **Defer to a dedicated CIP**; do not overload 0027.

## Phase B′ experiment matrix

On unitsquare (and optionally oil-class / crescent):

1. RBF + White + fixed probit (status quo softener)
2. RBF-only + fixed probit
3. RBF + White, no Bias + fixed probit
4. Optional spike: RBF-only + **trainable** probit \(\sigma^2\) (Option B), recorded as evidence for the future CIP — not a 0027 delivery gate

Metrics: classification error, active-set overlap vs (1), wall time for `selectPoints` + one outer iter, qualitative gnuplot margins.

**Decision rule for CIP-0027:** unless (2) or (4) clearly matches (1) on accuracy and selection stability, **IVM defaults keep White**. Bias auto-append follows B′ (likely drop if probit bias covers mean offset).
