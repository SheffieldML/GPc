---
id: "2026-11-19_cip002A-numerics-soak"
title: "CIP-002A numerics soak: oracles, invariants, clamp assert"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-21"
category: "features"
related_cips: ["002A", "0012", "0026"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- svgp
- numerics
- testing
- probit
- gradients
---

# Task: CIP-002A numerics soak (post-merge follow-up)

> After [PR #84](https://github.com/SheffieldML/GPc/pull/84) landed Phases 1–4 on main. Phase 1–3 gates already cover DTCVAR match, fixed point, monotone γ&lt;1, minibatch unbiasedness, M=N, FD-at-fixed-q, and seed reproducibility — do **not** re-gate those.

## Description

Add a focused numerics soak to `testSvgp` (and fixtures where needed) that closes the remaining assert-and-verify gaps for SVGP. Prefer hard oracles against existing code over weak “ELBO went up” checks.

### First five (do these)

1. **Hyperparameter gradients at q\***. At the Gaussian optimum from γ=1, SVGP ELBO grads w.r.t. kernel / Z / β (unconstrained packing) must match `CGp::logLikelihoodGradient` for DTCVAR at the same Z, kern, β. Envelope theorem: q\* contributes no extra term. Strongest Phase 2 check of the `CCholAdjoint` chain (including off-diagonal halving).
2. **Natural-gradient step for non-Gaussian.** For probit, the step’s target minus current natural parameters must equal the FD gradient of the ELBO w.r.t. expectation parameters \((\mathbf{m}, \mathbf{S}+\mathbf{mm}^\top)\). Verifies the CVI step for any likelihood once GH is in place.
3. **Exact probit marginal fixtures.** On small problems, ELBO ≤ exact log p(y) via Gaussian orthant probability (scipy-generated fixtures, same spirit as sparse PPA). Also compare SVGP-probit vs sparse PPA on shared problems.
4. **Nested inducing sets.** Gaussian: if Z ⊂ Z′, optimal ELBO(Z′) ≥ ELBO(Z). Direct variational-property gate.
5. **Clamp → assert.** Replace silent `var = 0` in `marginal()` with an assertion that any negative value is at most ~1e-10·k(x,x) before clamping; run extreme cases under it.

### Keep later (not first CI gates)

- Whitening KL identity on well-conditioned problems; Z-row permutation; d=2 = sum of two d=1; y/kernel/noise rescaling (−N d log s).
- GH vs MC / Jensen (E[log Φ] ≤ log E[Φ]); extreme tails |μ|/σ ∈ [10, 40].
- Adam+natgrad recovers CGp SCG hyperparameters; epoch sampler visits each index once; PD of S̃ under random γ≤1 for probit/ordered.
- GPy `SVGP` fixture (document whitened param map). One external fixture is enough before GPflow step-by-step.

### Defer / demote

- Second optimiser (L-BFGS vs natgrad) as a maintenance cost.
- Statistical 1/B gradient-variance check (flaky).
- N=1e5 timing smoke → `bench/` (CIP-0013 style).
- OpenMP bit-identity → REQ-0010 / CIP-001F.

## Acceptance Criteria

- [x] Gradients at q\* match `CGp` DTCVAR within a documented tolerance
- [x] Probit natgrad step matches FD on expectation parameters
- [x] Exact probit orthant fixtures: ELBO ≤ exact; SVGP vs sparse PPA compare documented
- [x] Nested inducing: ELBO(Z′) ≥ ELBO(Z) when Z ⊂ Z′ (Gaussian)
- [x] `marginal()` asserts near-nonnegativity before any clamp; extreme cases green under it
- [x] CIP-002A Testing Strategy updated with a “Numerics soak” subsection pointing here

## Implementation Notes

- Extend `testSvgp` (matio-free where possible); orthant fixtures as committed text/arrays like other GPy/PPA fixtures.
- Compare unconstrained param grads with the same transform convention as `CGp`.
- No-gradient items (4)–(5) and whitening/permutation can land before fixtures.

Landed 2026-11-21 on `cip002A-numerics-soak`: `testSvgp` soak section; `clampLatentVar` + `SANITYCHECK` in `CSvgp::fillMarginals*`. Orthant uses the closed-form bivariate Φ₂ identity (same as CIP-0030 Phase 3). Sparse PPA compare is Gaussian-site `ppaBound` on a shared problem (documented; not an orthant oracle).

## Related

- CIP: 002A
- REQ: 0015, 0002
- PRs: #82 (Phase 1), #84 (Phases 2–4)

## Progress Updates

### 2026-11-19
Task created after #84 merged; five-first list triaged from a broader numerics design note.

### 2026-11-21
First-five soak implemented and green in `testSvgp` (0 failures). Marked Completed.
