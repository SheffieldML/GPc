---
author: "Neil Lawrence"
created: "2026-11-18"
id: "001C"
last_updated: "2026-11-18"
status: "Implemented"
compressed: false
related_requirements: ["000D", "0002", "0005"]
related_cips: ["0012", "001B", "0013", "0017", "0019", "001A"]
tags:
- cip
- sparse
- inducing
- inter-domain
- white-noise
- numerics
- dtc
- dtcvar
- fitc
- pitc
title: "White-noise inducing features for stationary kernels (kernel-free K_uu)"
---

# CIP-001C: White-noise inducing features for stationary kernels (kernel-free \(K_{uu}\))

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented (Phase 1–2 on `cip001C-wn-pilot`)
- [x] Implemented - Phases 1–4 delivered (Hankel + stability cells); awaiting Close after PR/CI
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Introduce an **inducing-variable contract** in `CGp` (point inducing becomes one implementation) and add **white-noise inducing features**: write a stationary GP as \(f = g * w\) with \(w\) white noise and \(\hat g = \sqrt{S}\), and define inducing variables on the noise,
\[
u_m = \int \psi_m(\mathbf{x})\, w(\mathbf{x})\,\mathrm{d}\mathbf{x},
\]
with fixed windows \(\psi_m\). Then
\[
K_{uu}[m,n] = \int \psi_m\psi_n\,\mathrm{d}\mathbf{x}, \qquad K_{fu}(\mathbf{x},m) = (g*\psi_m)(\mathbf{x}).
\]
\(K_{uu}\) depends only on the windows; all kernel dependence moves into \(K_{fu}\). Hyperparameter learning can no longer push \(K_{uu}\) towards singularity, so no jitter is needed on it. The existing whitened StableFactor / LQ algebra ([CIP-0012](./cip0012.md), [CIP-001B](./cip001B.md)) is reused unchanged downstream of \((K_{uu}, K_{uf})\).

For RBF with Gaussian windows everything is closed form, and at window width \(s=\ell/\sqrt2\) the construction **reduces exactly to point inducing** (\(u_m = f(\mathbf{z}_m)/\sigma\)), which gives a sharp regression gate against the current code.

## Motivation

### The problem

Point inducing variables have \(K_{uu}=k(Z,Z)\). Its conditioning is governed by inducing spacing *in lengthscale units*; for RBF the spectrum decays super-exponentially, so moderate lengthscale growth with \(Z\) unchanged takes \(\kappa(K_{uu})\) from benign to Cholesky failure. Related routes to the same failure: ARD switching a dimension off (points distinct only in that dimension become duplicates), redundant inducing points drifting together once \(M\) exceeds what the kernel supports (the bound has no repulsion between coincident points), and \(\sigma_f^2\) growth shrinking fixed absolute jitter.

GPc's response so far is numerical: jitter in `_updateInvK` and whitening/LQ downstream. Those keep the bound computable but still factor \(K_{uu}\), and jitter is not free: adding \(\epsilon I\) to \(K_{uu}\) is exactly redefining \(u\) as noisy observations of \(f\), and the collapsed bound is monotonically decreasing in \(\epsilon\) (Burt, Rasmussen & van der Wilk 2020, Prop. 34). The same holds for any diagonal (per-inducing-point) jitter, by the same argument. Jitter is a pure loss to avoid, not a parameter to learn.

### The structural fix

Choose inducing variables whose prior covariance does not involve the kernel. White-noise features do this by construction. This is the classical inter-domain idea (Lázaro-Gredilla & Figueiras-Vidal 2009; Álvarez et al. 2010 variational inducing kernels for white-noise latent functions; Tobar et al. 2015 GPCM), used here purely as a conditioning device for ordinary stationary GPs.

Kernel-free \(K_{uu}\) does **not** remove singularity from coincident centres or from learning \(s\) into the point-inducing sweet spot; those remain complementary (cover trees / bounded \(s\)). This CIP removes *lengthscale-driven* failure of \(K_{uu}\).

### Equivalence with point inducing (RBF)

Isotropic RBF \(k(r)=\sigma^2\exp(-r^2/2\ell^2)\) in \(d\) dimensions. Its square-root kernel is \(g(\mathbf{x}) = \sigma(2\pi\ell^2)^{d/4}\,\mathcal{N}(\mathbf{x};\mathbf{0},\tfrac{\ell^2}{2}I)\) (so \(g*g=k\)). Take L²-normalised Gaussian windows
\[
\psi_m(\mathbf{x}) = (4\pi s^2)^{d/4}\,\mathcal{N}(\mathbf{x};\mathbf{z}_m,s^2 I).
\]
Then
\[
K_{uu}[m,n] = \exp\!\Big(-\frac{\lVert\mathbf{z}_m-\mathbf{z}_n\rVert^2}{4s^2}\Big),\qquad
K_{fu}(\mathbf{x},m) = \sigma\Big(\frac{2\sqrt2\,\ell s}{\ell^2+2s^2}\Big)^{d/2}\exp\!\Big(-\frac{\lVert\mathbf{x}-\mathbf{z}_m\rVert^2}{\ell^2+2s^2}\Big).
\]
(Checked numerically by direct convolution while drafting, 1-D, to ~1e-12.) The amplitude factor is the correlation between \(u_m\) and \(f(\mathbf{z}_m)/\sigma\) and is \(\le 1\) by AM–GM, with equality iff \(s=\ell/\sqrt2\), i.e. \(\psi_m\propto g(\mathbf{z}_m-\cdot)\). At that setting \(u_m=f(\mathbf{z}_m)/\sigma\) and \(K_{uu}=k(Z,Z)/\sigma^2\): **point inducing is the special case where the windows track the lengthscale**. Fixing \(s\) decouples them.

ARD is a product over dimensions with per-dimension \(\ell_j\) and \(s_j\). In GPc's RBF-ARD parameterisation \(\ell_j^2 = 1/(\gamma a_j)\) (`inverseWidth` \(\gamma\), `scales` \(a_j\)).

Log-derivatives for the isotropic case, with \(v=\ell^2+2s^2\) and \(r=\lVert\mathbf{x}-\mathbf{z}\rVert\):
\[
\partial_\ell \log K_{fu} = \tfrac d2\big(\tfrac1\ell-\tfrac{2\ell}{v}\big)+\tfrac{2\ell r^2}{v^2},\quad
\partial_s \log K_{fu} = \tfrac d2\big(\tfrac1s-\tfrac{4s}{v}\big)+\tfrac{4s r^2}{v^2},\quad
\nabla_{\mathbf{z}}\log K_{fu} = \tfrac{2(\mathbf{x}-\mathbf{z})}{v},
\]
and \(\partial_\sigma \log K_{fu}=1/\sigma\). These parallel RBF `psi1` (CIP-0017) with \(\ell\to\ell/\sqrt2\), \(S\to s^2\), plus the amplitude factor.

### Wasted capacity becomes measurable

In the collapsed bound GPc already factors \(C = I + \beta BB^\top\) with \(B=U^{-\top}K_{uf}\). The effective number of inducing features is
\[
M_{\text{eff}} = \operatorname{tr}\!\big(\beta BB^\top C^{-1}\big) = M - \operatorname{tr}(C^{-1}),
\]
which is cheap from `LcholC`. With point inducing, redundancy appears as ill-conditioning of \(K_{uu}\) (the matrix you cannot factor); with white-noise features it appears in the well-conditioned \(C\). The diagnostic is useful for both families and lands in Phase 4.

## Detailed Description

### Feasibility (codebase facts)

| Area | Today | Change |
|---|---|---|
| `CGp::_updateK` (~L1062) | `pkern->compute(K_uu, X_u)`, `pkern->compute(K_uf, X_u, *pX)` | Delegate to inducing object |
| Gradient block (~L1742–1780) | `gK_uu`, `gK_uf` → `getGradTransParams` / `getGradX` on `X_u` | Delegate to inducing object; for white-noise, `gK_uu` feeds window params only |
| Diagonal terms (FITC/DTCVAR `diagK`, `gLambda`; PITC `blockK`) | Kernel-only | Unchanged |
| Whitened / LQ algebra (`_whitenKuf`, `LcholK`, `LcholC`, adjoints) | Consumes \(K_{uu}, K_{uf}\) | Unchanged; may skip trsm when \(K_{uu}=I\) |
| Prediction `_testComputeKx` (L870) | `pkern->compute(kX, X_u, Xin)` | Delegate to inducing object |
| Params `getOptParams`/`setOptParams` (L631, L694) | Packs `X_u` | Inducing object packs its own params |
| I/O (matio L2690–2746; stream L2874–2905) | Asserts `X_u` cols == input dim | Inducing object owns shape checks; additive stream key |
| `CStationaryKern` | `K_of_r`, `scaleFactor` | Add optional `spectralDensity(rho)` (Phase 3) |

**Verdict:** feasible; roughly CIP-0012-sized. Phase 1 (contract refactor with no behaviour change) is the main risk because it touches every sparse code path; Phases 2–4 are additive. Phase 3 (Hankel) is the stretch item.

### Design principles (tenets)

1. **Compose Through Interfaces** — inducing variables become a contract; `CGp` never special-cases white-noise features. Mirrors GPflow's `Kuu`/`Kuf` dispatch on inducing-variable type.
2. **Assert and Verify** — the \(s=\ell/\sqrt2\) equivalence is an exact gate; FD on every new gradient; shape checks move into the inducing object.
3. **LAPACK Owns the Linear Algebra** — no new solver; StableFactor / LQ consume \((K_{uu},K_{uf})\) as today. \(K_{uu}\) and its Cholesky are cached and recomputed only when window parameters change.
4. **Flexibility Without a Performance Tax** — point-inducing path must not slow down (bench via CIP-0013 harness).
5. **Ecosystem Interoperability** — naming and the inducing-type split follow GPflow's inter-domain design so users recognise it.

### Phase 1 — Inducing-variable contract (no behaviour change)

Add `CInducing` (name indicative) with:

- `computeKuu(K_uu, kern)`, `computeKuf(K_uf, kern, X)`, `computeKux(kX, kern, Xstar)` (prediction)
- `gradKuuKuf(gParamsKern, gInducing, gX, kern, X, gK_uu, gK_uf, wantX)`: routes `gK_uu`/`gK_uf` to kernel params, inducing params and (if `isOptimiseX`) data inputs
- `kuuDependsOnKernel()`: lets `CGp` cache \(K_{uu}\) and `LcholK` across kernel-only parameter changes
- `getNumParams`, `getParams`/`setParams`, transforms, `toStream`/`fromStream`, `toMxArray`/`fromMxArray`, `numInducing()`, `domainDim()` (**not assumed equal to input dim**, so the weight-space follow-on can plug in)

`CInducingPoints` reproduces current behaviour exactly, moving the logic out of `CGp` lines cited above. `X_u` stays readable via an accessor for back-compat callers (`CGplvm` uses it once).

Gate: all existing reference, sparse, stability and dual-path tests bit-for-bit or within existing tolerances; CIP-0013 bench shows no regression.

### Phase 2 — White-noise features for RBF / RBF-ARD (Gaussian windows)

`CInducingWhiteNoise` with centres \(Z\) (\(M\times d\)) and window widths \(s\) (scalar, or per-dimension for ARD). Kernel-side cross-covariance is a new optional `CKern` virtual following the psi pattern (default throws `NotImplementedError`):

- `wnCrossCompute(Kfu, X, Z, s)`, `wnGradParams(g, X, Z, s, gKfu)`, `wnGradZ`, `wnGradS`, `wnGradX`

Implement in closed form for `CRbfKern`, `CRbfardKern`. `CCmpndKern`: all supported stationary components share one white-noise drive, so \(K_{fu}=\sum_c g_c*\psi_m\) and \(K_{uu}\) is unchanged; `CWhiteKern` contributes zero cross-covariance. `CBiasKern` (discrete spectrum, no L² root) throws with a message pointing to the non-goals.

\(s\) defaults to **fixed** (set from inducing spacing at initialisation). If made learnable it must be bounded above, otherwise it can drift towards \(\ell/\sqrt2\) and reintroduce the ill-conditioning.

### Phase 3 — Generic spectral path for isotropic `CStationaryKern`

Add `virtual double spectralDensity(double rho) const` (default throws) to `CStationaryKern`, implemented for `CMatern32Kern`, `CMatern52Kern`, `CExpKern` (closed-form Matérn densities). The radial profile of \(K_{fu}\) is the inverse Fourier transform of \(\sqrt{S(\rho)}\,\hat\psi(\rho)\): a 1-D Hankel transform of order \(d/2-1\), tabulated on an \(r\)-grid per hyperparameter evaluation and interpolated (a few hundred quadrature nodes). Lengthscale and variance gradients come from differentiating under the integral (same quadrature). Conventions are pinned by requiring the generic path, applied to RBF, to match the Phase 2 closed form.

`CRatQuadKern` is optional in this phase (its spectral density involves modified Bessel functions). ARD kernels other than RBF-ARD are out of scope (not isotropic, so the transform is not 1-D).

### Phase 4 — Diagnostics, stability cells, CLI, I/O

- `CGp::getEffectiveInducingCount()` returns \(M-\operatorname{tr}(C^{-1})\) (all inducing families)
- Numerical-stability suite (CIP-0019/001A): lengthscale-sweep cell with fixed \(Z\), comparing point vs white-noise on finiteness, jitter added, \(\kappa(K_{uu})\), ELBO/bound gap vs (jittered) point inducing, and FD agreement, under `stable` and `lq` modes
- `gp` CLI: long-form flags only, e.g. `--inducing points|whitenoise` and `--window-width s` (avoid short-flag collisions, cf. backlog `2026-11-15_gp-sparse-linalg-cli-collision`)
- Stream: additive `inducingType` key (absent = points); matio struct gains `inducingType`, `windowWidth`

### Explicit non-goals (this CIP)

- **Green's-function / causal-root features** (Brownian, integrated Brownian, causal Matérn \(g(t)=t^{p-1}e^{-\lambda t}\mathbb 1[t\ge0]\)); they give erf and smoothed-ReLU units in 1-D. Follow-on CIP, reusing this contract.
- **Weight-space white-noise features for dot-product kernels** (`CMlpKern` erf units; a new arc-cosine kernel for ReLU units; per-layer deep GPs; comparison with Dutordoir et al. 2021). Follow-on CIP; depends on `domainDim() != inputDim`.
- Psi statistics / Bayesian GP-LVM with white-noise features (closed form exists for RBF; follow-on).
- Orthonormal (box) windows giving \(K_{uu}=I\) exactly; tensor-product box windows for RBF-ARD are closed form (erf differences) and are a natural small extension once Phase 2 lands.
- Non-isotropic kernels other than RBF-ARD; periodic/cosine/bias kernels (no L² square root); learning \(s\) by default.

### Alternatives considered

1. **Learn jitter (shared or per point) as a variational parameter.** Rejected: valid bound, but monotone in the jitter, so the optimum is always the floor.
2. **RKHS-inner-product features with fixed shapes \(g_m\)** (VFF-style; Hensman et al. 2017, Dutordoir et al. 2020/2021). These give exact feature shapes, but \(K_{uu}=\langle g_m,g_n\rangle_{\mathcal H}\) depends on the kernel, so they do not meet REQ-000D. The white-noise features *are* RKHS features with \(g_m = L\psi_m\), parameterised from the noise side so the kernel leaves \(K_{uu}\).
3. **Variational Fourier / spherical-harmonic features.** Diagonal \(K_{uu}\) (benign even if badly conditioned), but they need a compact domain or the sphere and boundary terms for Matérn. Possible later implementations of the Phase 1 contract.
4. **Minimum-separation inducing selection** (Terenin et al., cover trees). Complementary: it addresses collapse of \(Z\), not lengthscale-driven conditioning. Could be an initialiser for either family.
5. **Hard-code a white-noise branch in `CGp`.** Rejected (Compose Through Interfaces); it would also block the weight-space follow-on.

## Implementation Plan

1. **Accept gate**: REQ-000D → Ready; create the four backlog tasks below. Pilot (`diagWhiteNoisePilot`) on `cip001C-wn-pilot` supplies lengthscale-sweep evidence before acceptance.
2. **Phase 1**: `CInducing` + `CInducingPoints`; move `X_u` logic out of `CGp`; all existing gates green; bench parity.
3. **Phase 2**: `CInducingWhiteNoise`; RBF/RBF-ARD closed forms and grads; compound rules; equivalence gate.
4. **Phase 3**: `spectralDensity` + Hankel path; Matérn 3/2, 5/2, Exp; RBF-through-generic-path gate.
5. **Phase 4**: \(M_{\text{eff}}\), stability-suite cells, CLI, I/O.
6. **Close**: compress into `docs/numerics.md` (sparse section) and README CLI notes.

## Backward Compatibility

- Point inducing remains the default; `.model` and matio files without `inducingType` load as points.
- `CKern` and `CStationaryKern` gain virtuals with throwing defaults; existing kernels compile unchanged.
- `CGp::X_u` access preserved via accessor during Phase 1; direct member use is migrated in-tree.
- No change to `--sparse-linalg` semantics.

## Testing Strategy

- **Equivalence (Phase 2):** with \(s=\ell/\sqrt2\) set per evaluation, log-likelihood matches `CInducingPoints` for DTC, DTCVAR, FITC, PITC under `legacy`, `stable` and `lq`. The likelihoods depend on \(u\) only through \(Q_{ff}=K_{fu}K_{uu}^{-1}K_{uf}\), which is invariant to invertible linear maps of \(u\). Gradient check uses the total derivative \(\partial_\ell L_{\text{wn}} + \tfrac{1}{\sqrt2}\partial_s L_{\text{wn}} = \mathrm{d}L_{\text{pt}}/\mathrm{d}\ell\); \(\sigma\), \(Z\), \(\beta\) gradients match directly.
- **FD:** `diagWhiteNoise` driver (pattern of `diagDtcvar`) for kernel params, \(Z\), \(s\), and \(X\) when optimised.
- **Kernel unit tests:** `wnCrossCompute` vs numerical convolution (1-D, 2-D), mirroring `testPsi`'s MC checks.
- **Generic path (Phase 3):** RBF via Hankel matches closed form; Matérn via Hankel matches brute-force 1-D/2-D convolution.
- **Stability (Phase 4):** lengthscale sweep cell: white-noise path finite with no \(K_{uu}\) jitter across the sweep; point path's jitter/failure recorded as the expected contrast (xfail-style where it breaks); ELBO/bound gap vs jittered point inducing recorded on the same cells.
- **Round-trip:** stream and matio save/load for both inducing types.

## Related Requirements

- [REQ-000D](../requirements/req000D_jitter-free-sparse-conditioning.md): primary outcome
- [REQ-0002](../requirements/req0002_runnable-verification.md): verification path
- [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md): kernels gaining `spectralDensity`

## Implementation Status

- [x] Phase 1: `CInducing` contract + `CInducingPoints`, no behaviour change
- [x] Phase 2: `CInducingWhiteNoise` + RBF/RBF-ARD closed forms + equivalence gate
- [x] Phase 3: `spectralDensity` + generic Hankel path (Matérn 3/2, 5/2, Exp); RBF-via-Hankel gate ~1e-10
- [x] Phase 4: Meff, CLI `--inducing`/`--window-width`, stream keys, `wn_ls_sweep_*` stability cells
- [ ] Compression into `docs/numerics.md`
- [x] Pilot spike (`diagWhiteNoisePilot` / `diagVikCompare`): pre-acceptance evidence


## Future CIPs (pointers, not scope)

- **Green's-function features (1-D):** → **[CIP-0029](./cip0029.md)** (Proposed). Causal spectral factor instead of \(\sqrt S\); Brownian gives \(\Phi((x-z)/s)\) (erf) units, integrated Brownian gives Gaussian-smoothed ReLU units, causal Matérn-\(\nu\) gives their exponentially tempered versions. \(K_{uu}\) is still \(\langle\psi_m,\psi_n\rangle\).
- **Weight-space features for dot-product kernels:** → **[CIP-0028](./cip0028.md)** (In Progress). Same white-noise pattern on weight space: \(f=\int\phi\sqrt{p}\,\xi\), \(u_m=\int\psi_m\xi\), so \(K_{uu}=\langle\psi,\psi\rangle\) and \(K_{fu}=\int\phi\sqrt{p}\,\psi\) (Gaussian product shifts window mean/cov by \(\Sigma\); early one-line erf sketch here omitted \(\sqrt{p}\)—see CIP-0028). Arc-cosine (order 1) gives smoothed ReLU ridge units.

## References

- M. Titsias (2009). *Variational learning of inducing variables in sparse Gaussian processes.* AISTATS.
- M. Lázaro-Gredilla & A. Figueiras-Vidal (2009). *Inter-domain Gaussian processes for sparse inference using inducing features.* NeurIPS.
- M. Álvarez, D. Luengo, M. Titsias & N. Lawrence (2010). *Efficient multioutput Gaussian processes through variational inducing kernels.* AISTATS.
- F. Tobar, T. Bui & R. Turner (2015). *Learning stationary time series using Gaussian processes with nonparametric kernels.* NeurIPS.
- J. Hensman et al. (2017). *Variational Fourier features for Gaussian processes.* JMLR.
- D. Burt, C. E. Rasmussen & M. van der Wilk (2020). *Convergence of sparse variational inference in Gaussian processes regression.* JMLR (Prop. 34: bound monotone in jitter).
- M. van der Wilk et al. (2020). *A framework for interdomain and multioutput Gaussian processes.* arXiv:2003.01115.
- V. Dutordoir, N. Durrande & J. Hensman (2020). *Sparse Gaussian processes with spherical harmonic features.* ICML.
- S. Sun, J. Shi & R. Grosse (2020). *Neural networks as inter-domain inducing points.* AABI.
- V. Dutordoir et al. (2021). *Deep neural networks as point estimates for deep Gaussian processes.* NeurIPS.
- A. Terenin et al. (2024). *Numerically stable sparse Gaussian processes via minimum separation using cover trees.* JMLR.
- [CIP-0012](./cip0012.md) whitened StableFactor; [CIP-001B](./cip001B.md) LQ; [CIP-0017](./cip0017.md) psi contract pattern; [CIP-0019](./cip0019.md)/[CIP-001A](./cip001A.md) stability suite.
