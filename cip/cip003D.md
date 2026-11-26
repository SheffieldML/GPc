---
author: "Neil Lawrence"
created: "2026-11-26"
id: "003D"
last_updated: "2026-11-26"
status: "Proposed"
compressed: false
related_requirements: ["0002"]
related_cips: ["0028", "0032", "0035", "0037", "003C"]
tags:
- cip
- deepgp
- weightspace
- inducing
- hermite
title: "Hermite ridge windows as a fixed weight-space inducing family"
---

# CIP-003D: Hermite ridge windows as a fixed weight-space inducing family

> CIPs describe HOW to achieve requirements (WHAT). REQ-0002 covers runnable verification.

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Code changes complete
- [ ] Closed - Verified and merged
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Add `CInducingHermite`, a second weight-space inducing family alongside the Gaussian ridge of CIP-0028, and expose it as `inducing='hermite'` on `DeepGP` (and the sparse GP façade). The prototype in `cip/cip0035/prototype/dgp.py` (`kind='hermite'`) supplies the maths and reference numbers; this CIP ports it to C++ with the gates GPc requires. It closes the last open item of CIP-003C, and the backlog task `2026-11-21_weightspace-hermite-ridges`.

## Motivation

The paper's Hermite variant cannot be run through GPc today: the only Hermite code in the library is Gauss–Hermite quadrature for non-Gaussian likelihoods. CIP-003C confirmed this when it tried to expose "Hermite features" on `DeepGP`.

There is also a numerical reason, independent of the paper. Gaussian ridge windows give a clean "smoothed erf neuron" mean, but \(\kappa(K_{uu})\) collapses when the window width \(s\) grows relative to offset spacing, at \(s=\sqrt2\), or at offset \(a=0\). CIP-0028 and CIP-0032 guard against this with bounds on \(s\) and one-sided offsets. Hermite windows have no width and no offset, so those failure modes do not arise; the backlog task reports \(\kappa\) of tens to hundreds in prototype sweeps.

The cost is the interpretation: a Hermite window is a polynomial of a projection, not a smoothed erf unit, so the finite-network reading of CIP-0028 is lost. We should say what the evidence supports. In the CIP-0035 composite task (three layers, hidden width 5), fixed Hermite windows beat fixed ridge windows at \(M=128\) (RMSE 0.315 against 0.623) but trail inducing points (0.238) and learned ridges (0.258). The case for the family is paper reproducibility and a well-conditioned \(K_{uu}\), not accuracy.

This aligns with the tenets "Compose Through Interfaces" (a new `CInducing` implementation, no change to `CSvgp` or `CDeepGp` logic) and "Assert Invariants and Verify Against Reference" (closed forms are gated against quadrature and against the prototype's recorded output).

## Detailed Description

### Maths (from the prototype)

Work in the MLP kernel's augmented input \(\hat{\mathbf x}=(\sqrt{w/D_\text{in}}\,\mathbf h,\sqrt b)\in\mathbb R^{D'}\), \(D'=d+1\), as the ridge family already does. A Hermite window is indexed by a unit direction \(\mathbf v\in\mathbb R^{D'}\) and an odd degree \(k\):

\[
\psi_{\mathbf v,k}(\boldsymbol\omega)=\frac{\mathrm{He}_k(\mathbf v^\top\boldsymbol\omega)}{\sqrt{k!}}\sqrt{\mathcal N(\boldsymbol\omega;0,I)} .
\]

- \(K_{uu}[m,n]=\delta_{k_mk_n}(\mathbf v_m^\top\mathbf v_n)^{k_m}\): independent of the kernel hyperparameters, so it is factorised once, as for the ridge.
- \(K_{fu}[i,m]=\sqrt{\sigma^2}\,c_{k_m}\,\rho_{im}^{k_m}\), with \(\rho_{im}=\mathbf v_m^\top\hat{\mathbf x}_i/\sqrt{1+2\|\hat{\mathbf x}_i\|^2}\) and \(c_k=\tfrac{2}{\sqrt\pi}(-1)^j\tfrac{(2j)!}{j!\sqrt{k!}}\), \(j=(k-1)/2\).
- Only odd degrees appear, because the erf kernel is odd in each argument.

### Layout (capacity-aware allocation)

Degree 1 takes `min(D', M)` windows with an orthonormal frame of directions (reusing the Gram–Schmidt blocks of `makeRidgeInit`). Remaining windows are dealt round-robin to degrees 3, 5, 7, 9 up to a per-degree capacity \(\lfloor\binom{D'+k-1}{k}/2\rfloor\), with random unit directions. `kmax=9` as in the prototype. If \(M\) exceeds total capacity, construction throws rather than truncating (assert, do not silently shrink).

### C++ interface

A new class `CInducingHermite : public CInducing` in `CInducing.h/.cpp`, not a subclass of `CInducingWeightSpace`: it has no \(s\), no offset, and no learnable parameters, so inheriting the ridge's width machinery would be misleading. State is `V` (M×D') and `ks` (M). `typeName()` is `"hermite"`; `kuuDependsOnKernel()` is false; `getNumParams()` is 0 (fixed windows, as in the prototype). `computeKuu` is kernel-free; `computeKuf` / `computeKux` use the MLP kernel's \(w\), \(b\), \(\sigma^2\) the same way the ridge hooks do; `gradFromKuuKuf` routes gradients to the kernel parameters only, and leaves `gZ` empty. It requires `CMlpKern`; any other kernel throws.

`CGp::setInducing("hermite")` and `CDeepGp::setLayerInducingType` gain the new name. Stream I/O gets a versioned `hermite` block (`V`, `ks`) so old `.model` files stay loadable.

### Python façade

`inducing='hermite'` on `DeepGP` (kernel `'mlp'` required, as for `'weightspace'`) and on `SparseGPRegression`. `learn_window_width(s)` is rejected for Hermite with an error that says why.

## Implementation Plan

1. Port `hermite_layout` and `c_k` to C++ (`CInducingHermite::makeLayout`); unit tests for allocation, capacity and the \(\kappa(K_{uu})\) range.
2. `computeKuu`, `computeKuf`, `computeKux`; gate each against quadrature or Monte Carlo over \(\boldsymbol\omega\), and against the prototype's recorded \(K_{uu}\) / \(K_{fu}\) summaries.
3. `gradFromKuuKuf` for kernel parameters; `checkGradients` on a sparse GP with Hermite inducing.
4. Wire into `CGp`, `CDeepGp`, stream I/O, CLI `--inducing hermite`.
5. Python façade, tests, and a short section in `docs/tutorials/deepgp.md` stating the trade-off against Gaussian ridges.
6. Reproduce the prototype's composite-task row for fixed Hermite windows (RMSE / NLL / ELBO per point) through the façade.

## Backward Compatibility

Purely additive: a new inducing type name and a new stream block. Defaults and existing `.model` files are unchanged.

## Testing Strategy

- \(K_{uu}\), \(K_{fu}\): compare with numerical integration over \(\boldsymbol\omega\) at small \(D'\), at several \(M\) and degrees.
- Gradients: `checkGradients` on `CGp` and on a `CDeepGp` layer, per the project's rule that gradients that disagree do not ship.
- Regression sweep over \(M\) and \(d\) (the lesson of CIP-003C item 1): construction succeeds or throws a named capacity error for every \((M,d)\), with no zero rows.
- Reference: reproduce the prototype's fixed-Hermite figures from CIP-0035 within stated tolerance; where they differ, say so in the CIP rather than loosening the test.
- Stream round-trip: write, read, and compare `K_{uu}` and predictions with `equals`.

## Related Requirements

- REQ-0002 (runnable verification)

## Open Questions

- Does `CMlpKern` already expose the \(\hat{\mathbf x}\) projections the Hermite \(K_{fu}\) needs, or do we add a hook? Check when starting phase 2.
- Should the per-degree capacity rule be a user option? The prototype hard-codes it; we propose to keep it fixed until a use case appears.
- Hermite in hidden layers: CIP-0035 lists it as "not recommended". We expose it anyway for reproduction and keep that note in the docs.

## Implementation Status

- [ ] Layout and \(c_k\) in C++
- [ ] \(K_{uu}\), \(K_{fu}\), \(K_{ux}\) with quadrature gates
- [ ] Kernel-parameter gradients and `checkGradients`
- [ ] `CGp` / `CDeepGp` / stream / CLI wiring
- [ ] Python façade and tests
- [ ] Docs: trade-off against Gaussian ridges
- [ ] Prototype composite-task reproduction

## References

- `cip/cip0035/prototype/dgp.py` (`hermite_layout`, `c_k`, `features`) and `dgp_wide.py`
- CIP-0028 (Gaussian ridge), CIP-0032 (per-unit widths), CIP-0035 (deep GP evidence table), CIP-0037 (`makeRidgeInit`), CIP-003C (façade gaps)
- Backlog: `backlog/features/2026-11-21_weightspace-hermite-ridges.md`
