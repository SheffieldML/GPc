---
id: "0014"
title: "Gaussian observation noise lives on the likelihood; no default White or Bias"
status: "Proposed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-20"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces", "assert-and-verify"]
stakeholders: ["maintainers", "modellers", "ppa-users", "gpy-porters"]
tags: ["noise", "white", "bias", "gaussian", "gpy", "gpy2", "architecture", "ppa", "ftc", "ivm", "deprecation"]
---

# REQ-0014: Gaussian observation noise lives on the likelihood; no default White or Bias

## Description

For Gaussian-process regression and for PPA’s Gaussian site layer, **observation noise is a property of the likelihood / noise model**. The canonical path is **`likelihood.variance`**; the GPy1 spelling **`Gaussian_noise.variance`** is an alias ([docs/target-architecture.md](../docs/target-architecture.md) rule 7). Callers must not be forced to put measurement noise in the prior via a default `CWhiteKern`. Default construction for those paths: **do not auto-append White or Bias**. `CWhiteKern` / `CBiasKern` remain available as **explicit** prior components.

**IVM exception:** classification softness that is **trained** today lives in White-in-\(K\) (probit \(\sigma^2\) is fixed and tiny). GPc may **retain trained White on the IVM path** after a documented composition check. Trainable probit scale, EP/Laplace on `CGp`, and GPy classification parity belong to a later **non-linear likelihoods** effort — not this requirement’s close gate.

**Why this matters**: Today FTC trains noise inside `white.variance` while sparse `CGp` and PPA use \(\beta\). CLI/demos also auto-add Bias. GPy’s defaults and advice put measurement noise in the likelihood and leave Bias/White opt-in. Aligning GPc removes double-counting, matches the ecosystem, and makes sparse↔dense / PPA↔`CGp` checks honest without a “strip White” special case.

**Who benefits**: Anyone porting GPy scripts; PPA / sparse-PPA (CIP-0023 / CIP-0026); Python façade (CIP-001D); tutorial and demo authors; IVM maintainers deciding kernel defaults with evidence.

## Acceptance Criteria

- [ ] FTC Gaussian regression trains and predicts with observation noise from `CGaussianNoise` at canonical path `likelihood.variance` (alias `Gaussian_noise.variance`; sparse algebra may keep internal \(\beta = 1/\sigma^2\))
- [ ] Every model that already exposes Gaussian observation noise / precision converges on that likelihood-owned parameter (not FTC-only)
- [ ] Default CLI / Python / demo constructions for Gaussian regression and PPA **do not** auto-append White or Bias
- [ ] When both a trainable White component and trainable likelihood variance are present, the library emits a clear deprecation or double-counting warning
- [ ] Dense and sparse PPA keep observation / site noise in \(\beta\) + `CNoise`; they do not depend on default White or Bias for correctness
- [ ] DTC \(M=N\), \(X_u=X\), signal kernel only matches dense PPA / FTC-with-noise-on-Gram without a White-kernel carve-out
- [ ] Old `.model` streams that embed White (and/or Bias) as part of the trained prior still load; migration / deprecation path is documented
- [ ] IVM: composition write-up confirms whether White remains the default softener; **preferred outcome is keep White**, with the exception documented in IVM tutorials / CLI help
- [ ] `CWhiteKern` / `CWhitefixedKern` / `CIndexWhiteKern` / `CBiasKern` and CIP-001C white-noise **inducing** features remain available; non-goals do not delete those types
- [ ] Non-goals: shipping trainable probit \(\sigma^2\) as the IVM softener; EP/Laplace/`CGp` classification parity; forced CIP-001C rename; silent drop of IVM White without the composition write-up

## Notes (Optional)

HOW: [CIP-0027](../cip/cip0027.md). Softener detail: [cip/cip0027/ivm_softener.md](../cip/cip0027/ivm_softener.md). Architecture: [docs/target-architecture.md](../docs/target-architecture.md). Related: [CIP-0026](../cip/cip0026.md).

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces, assert-and-verify
- **External**: GPy [#506](https://github.com/SheffieldML/GPy/issues/506); `GPRegression` / `GPClassification` defaults
- **Architecture**: [docs/target-architecture.md](../docs/target-architecture.md)

## Progress Updates

### 2026-11-18
Requirement drafted; CIP-0027 Proposed.

### 2026-11-18 (design lock)
Canonical path `Gaussian_noise.variance`; Bias auto-append in scope for regression/PPA (remove); IVM prefers **retain trained White**; EP / trainable probit deferred to a non-linear-likelihoods CIP.

### 2026-11-18 (IVM softener)
Write-up: White-in-\(K\) ≢ scaled probit; `computeElement` always 0 for White so IVM patches via `getWhite()`; B′ documents the exception rather than flipping IVM to RBF-only.

### 2026-11-20 (architecture amendment)
Canonical path becomes `likelihood.variance` with `Gaussian_noise.variance` as GPy1 alias; CIP-0027 Phase 0 adds batched likelihood API (no new state); `CNoise` data/site strip deferred to follow-on CIP.
