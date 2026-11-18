---
author: "Neil Lawrence"
created: "2026-11-18"
id: "001B"
last_updated: "2026-11-18"
status: "Accepted"
compressed: false
related_requirements: ["0002"]
related_cips: ["0012", "000E", "0013", "0019", "001A"]
tags:
- cip
- numerics
- sparse
- lq
- qr
- dtc
- dtcvar
- fitc
- vfe
- lapack
title: "LQ factorisation for collapsed VFE/DTC (and FITC) sparse bounds"
---

# CIP-001B: LQ factorisation for collapsed VFE/DTC (and FITC) sparse bounds

> CIPs describe HOW to achieve requirements (WHAT). This CIP extends [CIP-0012](cip0012.md)'s whitened StableFactor path by replacing the Gram+Cholesky of \(C = I + \beta BB^\top\) with an LQ (QR) factorisation of a wide stacked matrix — the same numerical trick Seeger, Hetzel, Dai, Meissner & Lawrence (2017) demonstrate for Bayesian linear regression, applied to the collapsed VFE/DTC bound which *is* BLR in disguise.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Keep CIP-0012's whitening \(A = \sigma^{-1} L_m^{-1} K_{mn}\) (equiv. \(B = U^{-\top}K_{uf}\) in GPc upper storage). **Stop** forming \(AA^\top\) and taking `potrf(I+AA^\top)`. Instead LQ-factor a wide matrix whose Gram *is* that quantity, optionally with a \(y\)-row so the Woodbury quadratic form falls out of the same factorisation. Gradients use the paper's LQ reverse-mode. Gate against StableFactor Cholesky and against a dense / mpmath reference on CIP-001A stress cells (crowded \(Z\), high \(\gamma\)).

## Motivation

CIP-0012 fixed the worst Legacy pathology (`pdinv` of \(K_{uu}\) / \(A\)) by whitening through \(L_m=\mathrm{chol}(K_{uu})\). The remaining Gram step still **squares the condition number** of \(A\):

\[
C = I + AA^\top \quad\text{then}\quad \mathrm{chol}(C).
\]

That is exactly the operation the Seeger et al. BLR example avoids with LQ of \([I,\,\alpha^{1/2}X]\). Collapsed VFE/DTC is Bayesian linear regression on whitened features \(A\), so the same trick carries over.

Evidence that the Gram step still hurts:

- CIP-0012 F3: DTCVAR LL errors remain large under StableFactor when \(\mathrm{cond}(K_{uu})\) is huge — the \(A\)-side / \(C\)-side, not only Legacy `pdinv`.
- CIP-001A stress (`sparse_dtcvar_crowdZ_*`): both Legacy and StableFactor show FD relative error \(\sim 0.89\) and dual-path LL gap \(\sim 10^{-3}\). Stable is “not worse,” not cured. Crowded \(Z\) makes \(K_{mm}\) hard; forming \(AA^\top\) then makes \(C\) harder still.
- MXNet/MXFusion sparse-GP demos (and GPc StableFactor) ship `potrf`+`trsm`+`syrk`+`potrf` — they never applied `gelqf` to the sparse bound, even though `gelqf` exists in the same operator set for BLR.

## Detailed Description

### 1. Bound as Bayesian linear regression

Notation: \(M\) inducing points \(Z\), \(N\) data, \(L_m=\mathrm{chol}(K_{mm})\) (map to GPc upper \(U\) via \(L_m\leftrightarrow U^\top\)), noise variance \(\sigma^2\) (GPc \(\beta=1/\sigma^2\)).

Whitened features:

\[
A = \sigma^{-1} L_m^{-1} K_{mn} \quad (M\times N).
\]

Then \(Q_{nn}+\sigma^2 I = \sigma^2(I+A^\top A)\). This is BLR with design \(\sigma A\), prior \(w\sim N(0,I)\), noise \(\sigma^2\). The Titsias / VFE bound is

\[
\mathcal{F}
= \log\mathcal{N}\bigl(y\,\big|\,0,\sigma^2(I+A^\top A)\bigr)
- \tfrac{1}{2\sigma^2}\operatorname{tr}K_{nn}
+ \tfrac12\|A\|_F^2.
\]

Plain DTC is the same without the trace correction. FITC replaces \(\sigma\) with heteroscedastic \(\Lambda^{1/2}\) (below).

### 2. LQ instead of Gram+Cholesky

As in Seeger et al. §3.3: \(I+AA^\top = [I,\,A][I,\,A]^\top\). Factor the wide matrix, do not form the Gram.

To obtain the data-fit term from the same factorisation, augment with \(y\):

\[
C=\begin{bmatrix} I_M & A\\ 0 & y^\top/\sigma\end{bmatrix}
\in\mathbb{R}^{(M+1)\times(M+N)},
\qquad
C=\tilde L\,\tilde Q,
\qquad
\tilde L=\begin{bmatrix} L & 0\\ c^\top & r\end{bmatrix}.
\]

Matching blocks of \(\tilde L\tilde L^\top=CC^\top\):

| Identity | Meaning |
|----------|---------|
| \(LL^\top=I+AA^\top\) | \(L\) is chol\((C)\) up to diagonal signs |
| \(Lc=Ay/\sigma\) | \(c=L^{-1}Ay/\sigma\) without a separate solve |
| \(r^2=\|y\|^2/\sigma^2-c^\top c\) | Woodbury quadratic \(y^\top(Q_{nn}+\sigma^2 I)^{-1}y\) |

Bound from one LQ:

\[
\mathcal{F}
= -\tfrac N2\log 2\pi - N\log\sigma
- \sum_i\log|L_{ii}|
- \tfrac12 r^2
- \tfrac{1}{2\sigma^2}\operatorname{tr}K_{nn}
+ \tfrac12\|A\|_F^2.
\]

If the \(y\)-row is omitted, \(c\) is still available as \(c=Q_2 y/\sigma\) with \(Q_2\) the last \(N\) columns of \(Q\) (since \(L^{-1}[I,\,A]=Q\)).

**Practical points (non-grad):**

- **Diagonal signs.** LAPACK `geqrf` / `gelqf` does not force \(\mathrm{diag}(L)\ge 0\). Use \(\log|L_{ii}|\) in the bound. For prediction reuse, flip row signs so the diagonal is positive; apply the **same** signed \(L\) in reverse mode (cotangent \(\bar L\) must flip with \(L\)).
- **Cost.** LQ is \(\mathcal{O}(M^2(M+N))\), same order as Gram+chol at roughly \(\sim 2\times\) flops; return is better \(\kappa\).
- **Storage.** GPc uses upper Cholesky elsewhere ([CIP-0005](cip0005.md)). Either store LQ \(L\) lower and convert at API boundaries, or implement against upper \(R\) from QR of \(C^\top\) consistently — pick one convention in Accept and document the Murray/Seeger \(\leftrightarrow\) GPc map once.

### 3. Gradients (reverse mode)

Do **not** hand-derive \(\partial\mathcal{F}/\partial C^{-1}\) in Legacy-style coordinates. Compose reverse mode through the forward graph, reusing CIP-0012 helpers for everything above the LQ.

#### 3.1 Computation graph

\[
K_{mm},K_{mn},\sigma
\;\xrightarrow{\mathrm{chol}}\;
L_m
\;\xrightarrow{\mathrm{trsm}}\;
A=\sigma^{-1}L_m^{-1}K_{mn}
\;\xrightarrow{\mathrm{stack}}\;
C=\begin{bmatrix}I&A\\0&y^\top/\sigma\end{bmatrix}
\;\xrightarrow{\mathrm{LQ}}\;
(\tilde L,\tilde Q)
\;\xrightarrow{\mathrm{unpack}}\;
(L,c,r)
\;\xrightarrow{\mathcal{F}}.
\]

Kernel / \(Z\) / \(\beta\) (or \(\sigma\)) cotangents exit through \(\bar K_{mm}\), \(\bar K_{mn}\), \(\bar\sigma\) into the existing `gK_uu` / `gK_uf` / `gLambda` / `gBeta` slots (same as CIP-0012).

#### 3.2 Scalar seed cotangents (spherical VFE / DTCVAR)

With \(\phi=\mathcal{F}\) (or \(-\mathcal{L}\) consistently with GPc’s LL sign), the only non-constant pieces that touch \((L,c,r)\) are \(\sum_i\log|L_{ii}|\) and \(\tfrac12 r^2\) (plus \(\tfrac12\|A\|_F^2\) and the trace term, which touch \(A\) / \(K_{nn}\) **outside** the LQ).

\[
\bar L_{ii} \;+=\; \frac{1}{L_{ii}}\operatorname{sgn}(L_{ii})
\quad\text{(no off-diagonal seed from the log-det)},
\qquad
\bar r \;+=\; r,
\qquad
\bar c \;+=\; 0
\quad\text{(from \(\tfrac12 r^2\) alone)}.
\]

If the implementation extracts \(c\) from \(\tilde L\) rather than from \(Q_2 y/\sigma\), there is still no direct \(\bar c\) from \(\mathcal{F}\) until prediction; the data-fit dependence on \(c\) is already inside \(r^2=\|y\|^2/\sigma^2-c^\top c\), so

\[
\bar c \;+=\; -c
\quad\text{(from \(-\tfrac12\cdot 2\,c\) when differentiating \(\tfrac12 r^2\) w.r.t. \(c\))}.
\]

(The two presentations are equivalent once \(r^2\) is written as a function of \(c\).) Trace / \(\|A\|_F^2\) contributions add \(\bar A \;+=\; A - \sigma^{-2}\partial(\operatorname{tr}K_{nn})/\partial A\) as in CIP-0012 DTCVAR (stable path); they do **not** go through the LQ adjoint.

Assemble \(\bar{\tilde L}\) by placing \(\bar L\) and \(\bar c\) (and \(\bar r\)) into the blocked lower factor; \(\bar{\tilde Q}\) starts at \(0\) unless a \(Q_2\)-based extraction of \(c\) was used (then \(\bar Q_2 \;+=\; \bar c\, y^\top/\sigma\)).

#### 3.3 LQ adjoint (Seeger et al. §4.3)

For wide full-rank \(C\in\mathbb{R}^{m\times n}\) (\(m=M{+}1\), \(n=M{+}N\)), \(C=\tilde L\,\tilde Q\):

\[
\bar C
=
\tilde L^{-\top}
\Bigl(
\bar{\tilde Q}
+
\mathrm{copyltu}(M)\,\tilde Q
\Bigr),
\qquad
M=\tilde L^\top\bar{\tilde L}-\bar{\tilde Q}\,\tilde Q^\top.
\]

Needs \(\tilde L\), \(\tilde Q\), and one temporary \(m\times m\) matrix. Implement as `_lqAdjoint` (or QR-of-transpose form with the same math). Unit-test against central differences on random wide matrices before wiring into `CGp`.

**Rank / crowded \(Z\):** the operator has no derivative at rank-deficient \(C\). Jitter / inducing separation policy stays with \(K_{mm}\); document that LQ reverse is undefined if `gelqf` fails or \(\lvert L_{ii}\rvert\) underflows.

#### 3.4 Unpack \(\bar C\) → \(\bar A\), \(\bar\sigma\)

Partition \(\bar C\) conformally with \(C\):

\[
\bar C
=
\begin{bmatrix}
\bar C_{11} & \bar C_{12}\\
\bar C_{21} & \bar C_{22}
\end{bmatrix}.
\]

Then \(\bar A \;+=\; \bar C_{12}\) (plus any direct \(\bar A\) from \(\|A\|_F^2\) / trace). The \(y\)-row block contributes to \(\bar\sigma\) only (fixed training \(y\)):

\[
\bar\sigma \;+=\;
\text{from }A=\sigma^{-1}(\cdots)\text{ and }y^\top/\sigma\text{ scalings}.
\]

No \(\bar y\) in the training objective.

#### 3.5 Chain through whitening (reuse CIP-0012)

With \(A=\sigma^{-1}L_m^{-1}K_{mn}\) (GPc: \(A\propto U^{-\top}K_{uf}\)):

1. **`trsm` adjoint** (existing `_trsmAdjoint`): \(\bar A\to(\bar K_{mn},\bar L_m)\) / \((\bar K_{uf},\bar U)\).
2. **Cholesky adjoint** (existing blocked Murray `_cholAdjoint`): \(\bar L_m\to\bar K_{mm}\).
3. **Noise / \(\beta\):** accumulate \(\bar\sigma\) (or \(\bar\beta\)) from the explicit \(\sigma^{-1}\) factors on \(A\) and on \(y/\sigma\), plus VFE trace \(\sigma\) dependence — same bookkeeping as CIP-0012’s \(\partial_\beta\) term, re-derived once for the LQ LL expression and checked by FD.

Do **not** duplicate a Gram-side `syrk` adjoint for \(I+AA^\top\); that node is gone.

#### 3.6 FITC / DTC / precision-form

| Approx | Extra reverse work |
|--------|-------------------|
| **DTC** | Drop trace / \(\|A\|_F^2\) DTCVAR terms; same LQ chain. |
| **FITC** | \(A\leftarrow A\Lambda^{-1/2}\), \(y\leftarrow\Lambda^{-1/2}y\). After \(\bar A\) on the scaled features, reverse the column scaling: \(\bar\Lambda\) from \(\bar A\) and \(\bar y_{\mathrm{scaled}}\), then \(\bar q\to\bar B\) as in CIP-0012 Appendix A. Write \(\bar\Lambda\) formulae in the implementing PR (or CIP appendix) before merge; mpmath/FD gate required. |
| **Precision form** | LQ input is \(\begin{bmatrix}L_m&K_{mn}/\sigma\end{bmatrix}\) (plus \(y\)-row). Reverse hits \(\bar L_m\), \(\bar K_{mn}\) from \(\bar C\), then chol adjoint on \(K_{mm}\). Trace cotangent is through \(\|L_m^{-1}L_\Sigma\|_F^2-M\) (one \(M\times M\) `trsm`), **not** an \(M\times N\) solve — see §5. |

PITC is **out of v1** unless Accept enlarges scope (block-diagonal \(\Lambda\) + shared LQ is a separate design beat).

#### 3.7 Code structure

- New: `_lqFactorWide`, `_lqAdjoint`, sign-normalise helper; cached \(\tilde L,\tilde Q\) (or \(L,c,r,Q_2\)) on the CIP-0009 AD cache.
- `gpCovGrads` / LL branch on `SparseLinalgLQ` after the StableFactor / Legacy branch.
- Reuse `_whitenKuf`, `_trsmAdjoint`, `_cholAdjoint` unchanged.
- NumPy prototype of §3.2–3.5 required before C++ grads merge (same bar as CIP-0012’s DTCVAR prototype).

### 4. Prediction

Posterior over whitened inducing values \(v=L_m^{-1}u\) is \(N(L^{-\top}c,\,(LL^\top)^{-1})\). With \(a=L_m^{-1}k_{m*}\):

\[
\mu_*=a^\top L^{-\top}c,
\qquad
\sigma^2_*=k_{**}-\|a\|^2+\|L^{-1}a\|^2.
\]

### 5. Variants

| Variant | Change |
|---------|--------|
| **Precision form** | LQ of \(\begin{bmatrix}L_m & K_{mn}/\sigma\\ 0 & y^\top/\sigma\end{bmatrix}\). Top block \(\Rightarrow L_\Sigma L_\Sigma^\top=K_{mm}+K_{mn}K_{nm}/\sigma^2\). Full VFE bound from this LQ + \(O(M^3)\) — including the trace term (below). |
| **FITC / heteroscedastic** | Replace \(\sigma\) by \(\Lambda^{1/2}\), \(\Lambda=\mathrm{diag}(K_{nn}-Q_{nn})+\sigma^2 I\). Scale column \(i\) of \(A\) and entry \(i\) of \(y\) by \(\Lambda_{ii}^{-1/2}\). Use \(\tfrac12\sum\log\Lambda_{ii}\) instead of \(N\log\sigma\); drop VFE trace. |
| **Plain DTC** | VFE without the trace correction \(\tfrac12\|A\|_F^2-\tfrac{1}{2\sigma^2}\operatorname{tr}K_{nn}\) pieces that distinguish VFE from DTC. |
| **\(K_{mm}\)** | Still needs chol (or jittered chol). If \(K_{mm}=\Phi\Phi^\top\) (e.g. RFF), LQ \(\Phi\) there too. |

#### 5.1 Precision-form trace term (no \(M\times N\) solve)

Earlier drafts claimed the VFE trace still required \(\|L_m^{-1}K_{mn}\|_F^2\) and thus an \(M\times N\) triangular solve. That overstates the cost. From the same precision-form LQ,

\[
\begin{bmatrix} L_m & K_{mn}/\sigma\\ 0 & y^\top/\sigma\end{bmatrix}
=
\begin{bmatrix} L_\Sigma & 0\\ \cdot & r\end{bmatrix}\tilde Q,
\qquad
L_\Sigma L_\Sigma^\top = K_{mm}+K_{mn}K_{nm}/\sigma^2.
\]

Define the \(M\times M\) matrix \(R=L_m^{-1}L_\Sigma\). Then

\[
RR^\top
= I + L_m^{-1}K_{mn}K_{nm}L_m^{-\top}/\sigma^2
= I+AA^\top = B,
\]

so \(R\) is a (non-triangular) square root of \(B\). Taking traces:

\[
\|A\|_F^2 = \operatorname{tr}(B)-M = \|L_m^{-1}L_\Sigma\|_F^2 - M.
\]

**From one LQ plus \(O(M^3)\):**

| Piece | Source |
|-------|--------|
| Log-det | \(\log\|B\| = 2\sum\log\|L_{\Sigma,ii}\|-2\sum\log L_{m,ii}\) |
| Data fit | \(r^2\) |
| Trace correction | \(\|L_m^{-1}L_\Sigma\|_F^2-M\), plus \(\operatorname{tr}K_{nn}\) (diagonal only) |

Cost: \(O(NM^2)\) for the LQ and \(O(M^3)\) for the \(M\times M\) `trsm` that forms \(L_m^{-1}L_\Sigma\). Everything that touches \(N\) is the single orthogonal factorisation — never \(L_m^{-1}K_{mn}\).

The same identity in the **whitened** form is \(\|A\|_F^2=\|L\|_F^2-M\); it saves nothing there because that path builds \(A\) explicitly anyway.

**Caveats:**

- **Cancellation.** \(\|A\|_F^2=\operatorname{tr}(B)-M\). When the data barely inform \(u\) (\(\|A\|_F^2\not\gg M\): large \(\sigma^2\), redundant inducing points), the subtraction loses digits. In typical regimes \(\|A\|_F^2\gg M\) and it is harmless.
- **Cannot drop the term.** It is what separates VFE from DTC. The identity only removes the expensive way of computing it.

### 6. Relationship to CIP-0012 / flag semantics

| Layer | CIP-0012 StableFactor (today) | CIP-001B (this) |
|-------|-------------------------------|-----------------|
| \(K_{uu}\) | chol \(U\) / \(L_m\) | unchanged |
| Whitening | \(B=U^{-\top}K_{uf}\) | same (\(A\propto B\)) |
| \(I+\beta BB^\top\) | `syrk` + `potrf` | **LQ of \([I,\,A]\) (or augmented \(C\))** |
| Grads | Murray chol/`trsm`/`syrk` adjoints | **LQ adjoint** (§3) then same whitening adjoints; no `syrk` adjoint on \(C\) |

**Accept / land choice:** introduce a third runtime mode `SparseLinalgLQ` (CLI `--sparse-linalg lq`) and **promote it to default** once forward+reverse and dual-path gates land. StableFactor remains `--sparse-linalg stable`. Do **not** silently change StableFactor's Gram+chol behaviour — LQ is a separate mode.

### 7. Non-goals

- Replacing Legacy (`pdinv`) archaeology.
- Multi-output / non-spherical sparse in v1 (same spherical scope as CIP-0012).
- Dropping \(K_{mm}\) jitter entirely — LQ does not remove the need for a well-posed \(K_{mm}\) factor when \(Z\) collapses.
- Porting MXNet `gelqf` operator wholesale; wrap LAPACK `gelqf`/`orglq` (or QR of transpose) in GPc/CMatrix style.

## Implementation Plan

1. **Prototype (offline).** Dense NumPy/SciPy: LQ bound vs CIP-0012 Cholesky bound vs full \(O(N^3)\) VFE on small \(N\); confirm \(\sim 10^{-13}\) agreement on well-conditioned cells (already sketched).
2. **LAPACK wrapper.** `CMatrix` (or thin helper) for LQ/QR of wide matrices; diagonal-sign normalisation helper; log-abs-diag.
3. **Forward path.** DTCVAR (VFE) first: build \(A\), augmented \(C\), extract \(L,c,r\), evaluate \(\mathcal{F}\). Then DTC, then FITC (\(\Lambda\) scaling).
4. **Prediction** matching §4.
5. **Reverse-mode grads** per §3: `_lqAdjoint` unit tests → DTCVAR seed cotangents → chain through existing `trsm`/`chol` adjoints → FITC \(\Lambda\) reverse in-PR; FD + dual-path vs StableFactor.
6. **CLI / enum** `--sparse-linalg lq`; document in [docs/numerics.md](../docs/numerics.md).
7. **Gates.** Extend `numerical_stability/` stress + `diagSparseStable` / mpmath: LQ vs StableFactor LL/grad; crowded-\(Z\) FD util; optional hard cell where Gram+chol degrades and LQ holds.

## Backward Compatibility

- Default is LQ (`SparseLinalgLQ` / `--sparse-linalg lq`); StableFactor and Legacy remain selectable.
- `.model` files still omit sparse-linalg mode (runtime only), same as CIP-0012.
- MatIO fixtures unchanged; LL must match StableFactor within CIP-001A budgets on mild cells.

## Testing Strategy

- Unit: LQ of \([I,\,A]\) vs `chol(I+AA^\top)` (abs diag); \(r^2\) vs Woodbury; sign-flip helper.
- Dual-path: LQ vs StableFactor on `testCGp` / `diagDtcvar` toys (\(\lvert\Delta\mathrm{LL}\rvert < 1\mathrm{e}{-10}\) mild).
- **Adjoint:** `_lqAdjoint` FD on random wide \(C\); DTCVAR \(\bar K_{uf},\bar K_{uu},\bar\beta\) FD + dual-path vs StableFactor (CIP-0012 prototype bar); FITC only with explicit \(\bar\Lambda\) formulae.
- Stress: CIP-001A crowded-\(Z\) and \(\gamma\times M\) slice — expect LQ FD / LL util not worse than StableFactor; document if still limited by \(K_{mm}\).
- Reference: mpmath recompute of bound from dumped \(K_{mm},K_{mn},y,\sigma\).
- CI: numerical-stability job + existing matio/CGp gates with `lq` optional matrix entry or env `GPC_SPARSE_LINALG=lq`.

## Implementation Status

- [x] Proposed design reviewed (Accept) — including §3 reverse-mode plan
- [x] LAPACK LQ/QR wrapper + sign policy (`dgelqf`/`dorglq`, `CLQAdjoint`, positive-diag normalise)
- [x] DTCVAR forward LL via LQ of \([I,\sqrt{\beta}B]\) (data-fit still via Legacy \(A^{-1}\) / StableFactor \(C\) consumers)
- [x] DTC + FITC variants (FITC: LQ for `LcholC`/`logDetC`; `Lm`/`bet` unchanged)
- [x] Prediction (reuses upper `LcholC` from LQ)
- [x] `_lqAdjoint` helper + unit tests (Seeger §4.3) — `testLQAdjoint`
- [x] DTCVAR/DTC reverse through LQ seeds + `lqAdjoint` → whitening adjoints; dual-path vs StableFactor in `testCGp`
- [x] FITC: \(C\)-block via `lqAdjoint` (then existing \(\bar D\) / \(\bar B\) chain); dual-path vs StableFactor
- [x] CLI mode + docs (`--sparse-linalg lq`, `docs/numerics.md`)
- [x] Ill-conditioned \(K_{uf}K_{fu}\) gate (`testLQAdjoint`): large-\(N\) clustered \(X\), OK \(K_{uu}\); LQ logdet beats Gram+chol vs SVD
- [x] Default promoted to LQ (`CGp` init + CLI `lq`)
- [ ] Stress / mpmath soak on GP LL/grad for the large-\(N\) clustered cell
- [ ] Crowded-\(Z\) remains \(K_{mm}\)-limited (LQ ≡ StableFactor there) — not the Seeger/C-side regime

**Accept notes (2026-11-18):** introduce `SparseLinalgLQ` (`--sparse-linalg lq`); GPc stores the LQ \(L\) as upper \(R=L^\top\) in `LcholC` for prediction reuse; PITC stays on the StableFactor Gram path under `lq` in v1. **Land notes:** LQ is the default sparse linalg mode; StableFactor stays available for dual-path soak.

## References

- M. Seeger, A. Hetzel, Z. Dai, E. Meissner, N. D. Lawrence (2017). *Auto-differentiating linear algebra.* arXiv:1710.08717 — §3.2 sparse GP (Cholesky whitening); §3.3 BLR (LQ of \([I,\alpha^{1/2}X]\)); §4.3 LQ forward/backward.
- Titsias (2009). Variational learning of inducing variables in sparse GPs — collapsed bound \(\mathcal{F}\).
- [CIP-0012](cip0012.md) — whitened StableFactor + Murray/Seeger chol adjoints (prerequisite).
- [CIP-000E](cip000E.md) — original StableFactor flag.
- [CIP-0013](cip0013.md) / [CIP-001A](cip001A.md) — bench and stress evidence that Gram-side errors remain.
- Quiñonero-Candela & Rasmussen (2005) — DTC / FITC taxonomy (not to be confused with LAPACK QR).

## Author and Date

Neil Lawrence, 2026-11-18.
