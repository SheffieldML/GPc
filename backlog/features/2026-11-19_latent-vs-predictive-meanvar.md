---
id: "2026-11-19_latent-vs-predictive-meanvar"
title: "Split posteriorMeanVar into latent and predictive (noisy) variants, GPy-style"
status: "Abandoned"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0031", "002A", "0027", "001D", "0026", "0023"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- prediction
- api
- gpy
- noise
- consistency
- superseded
---

# Task: Split `posteriorMeanVar` into latent and predictive variants

> **Abandoned (2026-11-21):** design promoted to [CIP-0031](../../cip/cip0031.md) (Proposed). Implementation backlog tasks spawn when that CIP is Accepted — do not implement from this file.
>

> Surfaced while verifying the [CIP-002A](../../cip/cip002A.md) SVGP prototype against `CGp` DTCVAR: predictive means agreed to 1e-14 but variances differed by exactly \(1/\beta\), because `CGp::posteriorMeanVar` adds observation noise on sparse paths while `CSvgp::posteriorMeanVar` returns the latent variance.

## Description

`posteriorMeanVar` means different things in different classes. GPy avoids this with two entry points: `predict(X, include_likelihood=True)` (distribution of \(y_*\)) and `predict_noiseless(X)` (distribution of \(f_*\)). GPc should do the same, with explicit names and one meaning each:

- `latentMeanVar(mu, varSigma, X)` — mean and variance of \(f_*\); no observation noise; GPy `predict_noiseless` / `_raw_predict`.
- `predictiveMeanVar(mu, varSigma, X)` — mean and variance of \(y_*\) under the likelihood; for Gaussian noise adds the noise variance once; GPy `predict`. For non-Gaussian likelihoods return moments from `noise.out` (or throw until defined).

Keep `posteriorMeanVar` temporarily as a deprecated alias with today's per-class behaviour, migrate internal callers, then remove.

### Audit (main @ 2026-11-19)

| Class | `posteriorMeanVar` variance today | Notes |
|---|---|---|
| `CGp` FTC | latent, **but** \(k(x,x)\) includes `CWhiteKern` variance when present | so "noise" arrives via the kernel; changes when [CIP-0027](../../cip/cip0027.md) drops auto-White |
| `CGp` DTC/DTCVAR/FITC/PITC | latent **+ \(1/\beta\)** (`_posteriorVar`, both whitened and legacy branches) | FTC and sparse disagree within one class |
| `CBayesGplvm` | latent **+ \(1/\beta\)** | |
| `CPpa` (dense) | latent (no \(1/\beta\)) | MATLAB `ppaPosteriorMeanVar` port |
| `CSparsePpa` | delegates to inner `CGp` ⇒ latent **+ \(1/\beta\)** | dense vs sparse PPA predictive variance differs by \(1/\beta\) before `outerNoise.out` |
| `CSvgp` (prototype) | latent | |
| `CIvm` | latent (no noise term found) | confirm |

Follow-on inconsistencies to resolve in this task:

1. `CGp::out(yPred, probPred, X)` passes the noisy sparse variance to `CGaussianNoise::out`, which adds `sigma2` again ⇒ noise counted twice (negligible at the default `sigma2 = 1e-6`, wrong if `sigma2` is learnt).
2. PPA: \(f=\bar f+\varepsilon\), \(\varepsilon\sim\mathcal N(0,1/\beta)\), so the input to the probit should arguably be \(\mathrm{Var}[\bar f_*]+1/\beta\). Dense `CPpa` omits \(1/\beta\); sparse includes it. Decide which is intended (check MATLAB `ppaOut`) and make both agree.

## Acceptance Criteria

- [ ] `latentMeanVar` and `predictiveMeanVar` exist on `CGp`, `CBayesGplvm`, `CMrd`, `CPpa`, `CSparsePpa`, `CIvm`, `CSvgp` with documented semantics
- [ ] `CGp` FTC and sparse paths agree: same latent variance at \(M=N\), \(Z=X\) (no White), and predictive = latent + \(1/\beta\)
- [ ] `out(…, probPred, …)` counts observation noise exactly once
- [ ] Dense and sparse PPA give the same predictive variance at \(M=N\) (no White); semantics documented
- [ ] `posteriorMeanVar` marked deprecated (comment + docs), internal callers migrated; removal tracked separately
- [ ] Python façade exposes `predict(X, include_likelihood=True)` and `predict_noiseless(X)` mapping to the two C++ calls (CIP-001D)
- [ ] Tests: latent vs predictive differ by exactly the Gaussian noise variance; `testSvgp` compares `CSvgp::latentMeanVar` to `CGp::latentMeanVar` without the manual \(1/\beta\) correction

## Implementation Notes

- Implement `latentMeanVar` in `CGp::_posteriorVar` by factoring out the `+ 1.0/getBetaVal()` lines; `predictiveMeanVar` adds noise via the noise model (Gaussian: \(1/\beta\) now, `Gaussian_noise.variance` after CIP-0027).
- FTC with `CWhiteKern`: latent variance should exclude the White term once CIP-0027 lands; until then document that White-in-kernel is treated as latent.
- Output scale/bias: apply in both variants as today (`scaleCol` by `scale^2`).

## Related

- CIP: 002A (SVGP prototype finding), 0027 (noise in likelihood), 001D (Python façade), 0023 / 0026 (dense / sparse PPA)

## Progress Updates

### 2026-11-19

Task created from the CIP-002A prototype comparison; audit table above from `main` at merge time.

### 2026-11-21

Promoted to CIP-0031. Status → Abandoned; keep file as historical design source linked from the CIP.
