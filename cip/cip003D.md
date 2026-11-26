---
author: "Neil Lawrence"
created: "2026-11-26"
id: "003D"
last_updated: "2026-11-26"
status: "In Progress"
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
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
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

### Ask versus delivered: errors and warnings

Principle: if what we build differs from what the caller asked for, we say so. We never adjust a request silently, and we never catch our own assertion to continue with a default (project tenet: assert invariants and verify against reference). Two classes:

**Errors (throw; nothing is adjusted).**

| Id | Condition | Error |
|---|---|---|
| E1 | `M` exceeds total capacity for \(D'\) | `ndlexceptions::Error` naming `d`, `M`, total capacity and per-degree capacities |
| E2 | Kernel is not `CMlpKern` / `CMlpardKern` (including a compound with no MLP signal) | `NotImplementedError` naming the kernel type |
| E3 | Window-width options (`window_width`, `learn_window_width(s)`) passed with Hermite | `Error` saying Hermite windows have no width |
| E4 | `inducing_fixed=False` passed *explicitly* with Hermite (windows have no free parameters) | `Error`; if the argument is left at its default, Hermite resolves to fixed and `str(model)` says so |
| E5 | Post-construction audit fails: delivered `numInducing() != M`, any degree even or below 1, any \(\mathbf v\) not unit norm, or the \(K_{uu}\) block Cholesky fails | `Error` (an internal-invariant assert, kept in release builds because the checks are \(O(M^2)\) once at construction) |
| E6 | Stream load: `V` rows, `ks` length and `numActive` disagree, or `ks` invalid | `StreamFormatError` |

**Warnings (proceed, but loudly).** C++ records a `HermiteLayoutReport` on the object (degree histogram, fill fraction per degree, \(\kappa(K_{uu})\), captured-variance fraction on the training inputs) and writes the warnings to `cerr` when verbosity is at least 1, as `CGp::maybeWarnDualObservationNoise` does. The Python façade reads the report and calls `warnings.warn` unconditionally, because stderr from C++ is easy to miss in a notebook.

| Id | Condition | Warning |
|---|---|---|
| W1 | \(\kappa(K_{uu})>10^4\) (the CIP-0028 budget) | conditioning, with the value |
| W2 | Total fill above 90% of capacity | `M` is near the capacity limit; conditioning degrades |
| W3 | Captured-variance fraction on the training inputs below a threshold | windows capture little of the prior variance (the CIP-0028 low-dimension plateau). The threshold is a placeholder at 0.5; phase 6 calibrates it against the prototype sweeps |
| W4 | Hermite used in a hidden layer of a multi-layer `DeepGP` | CIP-0035 finds fixed windows underfit in hidden layers; allowed for reproduction |

`describe()` / `str(model)` always show the delivered layout (degree histogram, \(\kappa\), captured variance) so a user who never reads warnings can still see what they got.

### C++ interface

A new class `CInducingHermite : public CInducing` in `CInducing.h/.cpp`, not a subclass of `CInducingWeightSpace`: it has no \(s\), no offset, and no learnable parameters, so inheriting the ridge's width machinery would be misleading. State is `V` (M×D') and `ks` (M). `typeName()` is `"hermite"`; `kuuDependsOnKernel()` is false; `getNumParams()` is 0 (fixed windows, as in the prototype). `computeKuu` is kernel-free; `computeKuf` / `computeKux` use the MLP kernel's \(w\), \(b\), \(\sigma^2\) the same way the ridge hooks do; `gradFromKuuKuf` routes gradients to the kernel parameters only, and leaves `gZ` empty. It requires `CMlpKern`; any other kernel throws.

`CGp::setInducing("hermite")` and `CDeepGp::setLayerInducingType` gain the new name. Stream I/O gets a versioned `hermite` block (`V`, `ks`) so old `.model` files stay loadable.

### Python façade

`inducing='hermite'` on `DeepGP` (kernel `'mlp'` required, as for `'weightspace'`) and on `SparseGPRegression`. `learn_window_width(s)` is rejected for Hermite with an error that says why.

### Phase 1 measurement: conditioning against fill (2026-11-26)

`\(\kappa(K_{uu})\)` over 20 seeds per cell, from `CInducingHermite::makeLayout`. The backlog task's "tens to hundreds" holds only up to about half capacity:

| d | fill | geometric-mean \(\kappa\) | worst of 20 |
|---|---|---|---|
| 1 | 50% (M=8) | 11 | 239 |
| 1 | 75% (M=12) | 78 | 3.5e5 |
| 1 | 100% (M=16) | 2.0e3 | 3.7e7 |
| 2 | 50% (M=31) | 157 | 1.9e3 |
| 2 | 75% (M=47) | 1.1e3 | 2.1e4 |
| 2 | 100% (M=63) | 5.9e3 | 1.4e5 |
| 3 | 50% (M=106) | 586 | 1.2e4 |
| 3 | 100% (M=212) | 6.4e3 | 2.9e4 |

Consequences. (1) The halved capacity keeps every layout positive definite (no E5 in the \(M=1..300\) sweep at \(d\in\{1,2,8\}\)), but it does not keep \(\kappa\) small near the limit. (2) W1 (\(\kappa>10^4\)) is therefore the operative guard; W2 at 90% fill fires after \(\kappa\) has already reached \(10^3\) in low dimensions. We keep both, because W2 is cheap and explains *why* W1 fired, and we leave the thresholds as agreed. (3) The d=1 worst case (3.7e7) is a single random draw of five close angles, so a deterministic, well-spread direction rule for \(D'=2\) would remove it; this is a candidate refinement, not part of this CIP, since the prototype's rule is random.

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
- Ask-versus-delivered: one test per error E1-E6 (each must throw, with the named fields in the message) and per warning W1-W4 (each must fire from a constructed case, and must not fire on a well-conditioned control). A test also asserts that `numInducing()` equals the requested `M` whenever construction succeeds.

## Related Requirements

- REQ-0002 (runnable verification)

## Open Questions (resolved or narrowed 2026-11-26)

**Kernel hooks (resolved: new hooks needed).** The existing hooks are ridge-specific virtuals on `CKern` (`wsRidgeCrossCompute`, `wsRidgeGradParams`, `wsRidgeGradX`, taking `V, a, s`), implemented for `CMlpKern`, `CMlpardKern`, `CArcCosineKern`, `CWhiteKern` and the compound dispatcher. The \(\hat{\mathbf x}=\sqrt{\sigma_\text{diag}}\,(\mathbf x,1)\) projection is formed inside file-local helpers (`ridgeKfuFillState`), not exposed. Phase 2 therefore adds parallel virtuals `wsHermiteCrossCompute` / `wsHermiteGradParams` / `wsHermiteGradX` (taking `V, ks`) on `CKern` (default: throw `NotImplementedError`, as the ridge hooks do), implemented for `CMlpKern` and `CMlpardKern`, and routed through the compound dispatcher. `CArcCosineKern` stays unsupported until asked for.

**Convention mismatch (new, must be handled in phase 2).** The prototype kernel is \(\sigma^2\tfrac2\pi\arcsin\!\big(2\hat x\!\cdot\!\hat x'/\sqrt{(1+2\|\hat x\|^2)(1+2\|\hat x'\|^2)}\big)\). GPc's `CMlpKern::computeElement` is \(\sigma^2\arcsin\!\big(\hat x\!\cdot\!\hat x'/\sqrt{(1+\|\hat x\|^2)(1+\|\hat x'\|^2)}\big)\): no \(2/\pi\) prefactor, and the factor 2 is absorbed in the weight and bias variances. So \(\rho\), \(c_k\) and the \(\sqrt{\sigma^2}\) prefactor above cannot be copied from the prototype; they must be re-derived for GPc's form (the ridge code's \(B=\sqrt{\sigma^2\pi/2}\) shows the same adjustment). The quadrature gate in phase 2 is what settles this; the prototype's figures are compared only after rescaling.

**Convention derivation (phase 2, gated).** Write \(\hat x=(\sqrt w\,x,\sqrt b)\), \(n_2=\|\hat x\|^2\), \(B=\sqrt{\sigma^2\pi/2}\). The identity \(\mathbb E[\operatorname{erf}(a\!\cdot\!\omega)\operatorname{erf}(b\!\cdot\!\omega)]=\tfrac2\pi\arcsin\big(2a\!\cdot\!b/\sqrt{(1+2\|a\|^2)(1+2\|b\|^2)}\big)\) with \(a=\hat x/\sqrt2\) gives \(k(x,x')=B^2\,\mathbb E_\omega[\operatorname{erf}(\omega\!\cdot\!\hat x/\sqrt2)\operatorname{erf}(\omega\!\cdot\!\hat x'/\sqrt2)]\), so \(K_{fu}=B\,\mathbb E_\omega[\operatorname{erf}(\omega\!\cdot\!\hat x/\sqrt2)\,He_k(v\!\cdot\!\omega)]/\sqrt{k!}\). For jointly Gaussian \((s,y)=(v\!\cdot\!\omega,\hat x\!\cdot\!\omega)\) with \(\operatorname{Var}s=1\), \(\mathbb E[He_k(s)h(y)]=\operatorname{Cov}(s,y)^k\,\mathbb E[h^{(k)}(y)]\); with \(h^{(k)}(y)=\sqrt{2/\pi}\,He_{k-1}(y)e^{-y^2/2}\) (\(k\) odd) and \(\mathbb E_{y\sim N(0,v)}He_{2m}(y)=(2m-1)!!\,(v-1)^m\), this reduces to

\[
K_{fu}[n,m]=\sigma\,c_k\,\rho^k,\qquad \rho=\frac{v\!\cdot\!\hat x}{\sqrt{1+n_2}},\qquad c_k=\frac{(-1)^j\,(2j-1)!!}{\sqrt{k!}},\ j=\tfrac{k-1}2 .
\]

The prototype's \(c_k=\tfrac{2}{\sqrt\pi}(-1)^j\tfrac{(2j)!}{j!\sqrt{k!}}\) and \(\rho_p=v\!\cdot\!\hat x_p/\sqrt{1+2\|\hat x_p\|^2}\) are the same expression for its kernel (\(\tfrac2\pi\sigma^2\arcsin\), \(\operatorname{erf}(\omega\!\cdot\!\hat x_p)\)): substituting \(\hat x_p=\hat x/\sqrt2\) gives \(\rho_p=\rho/\sqrt2\) and \(\sqrt{2/\pi}\,\sigma\) times our \(c_k\rho^k\), the \(\sqrt{2/\pi}\) being the \(2/\pi\) kernel prefactor. Implemented in `mlpHermiteKfuDiag` (`CKern.cpp`) for `CMlpKern` and `CMlpardKern` (ARD: \(\sigma_{\text{diag},j}=w\,\text{scale}_j\), bias last). Gates in `testInducingHermite`: the erf-feature identity reproduces `computeElement` (\(\lesssim10^{-11}\)), \(K_{fu}\) matches tensor-grid quadrature of the window integral (\(\lesssim10^{-11}\) for isotropic \(d=1\) and ARD \(d=2\)), and the Nyström capture never exceeds the prior variance (0.98 and 0.99 of it at \(M=9\)). The numerical comparison with the prototype's own output is left to phase 6; here the reconciliation is analytic.

**Per-degree capacity (decision, agreed 2026-11-26: keep fixed for now, store `ks` in the stream; over-capacity and near-capacity are error E1 and warning W2).** The rule is \(\lfloor\binom{D'+k-1}{k}/2\rfloor\) per odd degree \(k\le9\), degree 1 capped at \(D'\). Implications:

| input dim \(d\) | total capacity (\(k\le 9\)) | without the \(/2\) |
|---|---|---|
| 1 | 16 | 30 |
| 2 | 63 | 125 |
| 3 | 212 | 420 |
| 5 | 1557 | 3108 |
| 8 | 16106 | 32206 |

- *Low-dimensional inputs run out.* A 1-D problem (the sinc tutorials) cannot use more than 16 Hermite windows, a 2-D problem more than 63. The paper-scale \(M=64,128\) with hidden width 5 (\(D'=6\), capacity 1557) is unaffected. We assert on this: `M` above capacity throws a named error (E1). The prototype instead leaves its allocation loop with `rest > 0` and silently returns fewer than `M` windows (`self.M = len(ks)`), so reference runs at low \(d\) may have used fewer windows than requested; check before comparing.
- *The \(/2\) is a conservative margin, not a theorem.* Distinct non-parallel unit directions give a positive-definite degree-\(k\) block up to \(\binom{D'+k-1}{k}\) (the dimension of symmetric \(k\)-tensors), so the true rank bound is twice the stated capacity. Random directions near the true bound give a poorly conditioned block; halving keeps \(\kappa(K_{uu})\) in the tens to hundreds. We have not derived the factor, so the CIP should test \(\kappa\) against fill fraction rather than assert it.
- *High degrees carry little variance.* \(K_{fu}\propto c_k\rho^k\) with \(|\rho|<1\), so degree 7 and 9 windows add little unless \(\|\hat x\|\) is large. Above a few hundred windows in low \(d\), extra \(M\) buys little captured variance; this is the low-dimension plateau CIP-0028 describes for fixed windows.
- *Fixed means reproducible.* With a seed, the layout is deterministic and matches the prototype's design, which is what the reproduction gate wants. Users cannot trade degree mix against conditioning.
- *Deferral is cheap if `ks` is stored.* The stream block holds `V` and `ks` explicitly, so a later release can change `kmax` or the allocation policy (and expose it as an option) without breaking old `.model` files. If we omitted `ks` and recomputed it from `(M, D')`, any later change to the rule would silently change loaded models.

## Implementation Status

- [x] Layout, capacity rule, audit (E1, E5) and layout report (W1, W2) in C++ (`CInducingHermite`, phase 1)
- [ ] Ask-versus-delivered: E1, E2, E5, W1, W2 done and tested (phases 1-2); E3, E4, E6, W3, W4 follow in phases 4, 5
- [x] \(K_{uu}\), \(K_{fu}\), \(K_{ux}\) with quadrature gates (phase 2; `wsHermiteCrossCompute` on `CMlpKern`, `CMlpardKern`, compound; the `wsHermiteGradParams` / `wsHermiteGradX` virtuals arrive with phase 3)
- [ ] Kernel-parameter gradients and `checkGradients`
- [ ] `CGp` / `CDeepGp` / stream / CLI wiring
- [ ] Python façade and tests
- [ ] Docs: trade-off against Gaussian ridges
- [ ] Prototype composite-task reproduction

## References

- `cip/cip0035/prototype/dgp.py` (`hermite_layout`, `c_k`, `features`) and `dgp_wide.py`
- CIP-0028 (Gaussian ridge), CIP-0032 (per-unit widths), CIP-0035 (deep GP evidence table), CIP-0037 (`makeRidgeInit`), CIP-003C (façade gaps)
- Backlog: `backlog/features/2026-11-21_weightspace-hermite-ridges.md`
