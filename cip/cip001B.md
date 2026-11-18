---
author: "Neil Lawrence"
created: "2026-11-18"
id: "001B"
last_updated: "2026-11-18"
status: "Proposed"
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
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
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

**Practical points:**

- **Diagonal signs.** LAPACK `geqrf` / `gelqf` does not force \(\mathrm{diag}(L)\ge 0\). Use \(\log|L_{ii}|\) in the bound. For prediction reuse, flip row signs so the diagonal is positive.
- **Gradients.** Reverse-mode through the paper's LQ backward (needs \(L\), \(Q\), one \(M\times M\) temporary). Extend [CIP-0012](cip0012.md) adjoint helpers; do not hand-derive a third gradient path in coordinates of \(C^{-1}\).
- **Cost.** LQ is \(\mathcal{O}(M^2(M+N))\), same order as Gram+chol at roughly \(\sim 2\times\) flops; return is better \(\kappa\).
- **Storage.** GPc uses upper Cholesky elsewhere ([CIP-0005](cip0005.md)). Either store LQ \(L\) lower and convert at API boundaries, or implement against upper \(R\) from QR of \(C^\top\) consistently — pick one convention in Accept and document the Murray/Seeger \(\leftrightarrow\) GPc map once.

### 3. Prediction

Posterior over whitened inducing values \(v=L_m^{-1}u\) is \(N(L^{-\top}c,\,(LL^\top)^{-1})\). With \(a=L_m^{-1}k_{m*}\):

\[
\mu_*=a^\top L^{-\top}c,
\qquad
\sigma^2_*=k_{**}-\|a\|^2+\|L^{-1}a\|^2.
\]

### 4. Variants

| Variant | Change |
|---------|--------|
| **Precision form** | LQ of \(\begin{bmatrix}L_m & K_{mn}/\sigma\\ 0 & y^\top/\sigma\end{bmatrix}\). Top block \(\Rightarrow L_\Sigma L_\Sigma^\top=K_{mm}+K_{mn}K_{nm}/\sigma^2\). Log-det: \(2N\log\sigma+2\sum\log|L_{\Sigma,ii}|-2\sum\log L_{m,ii}\). Trace term still needs \(\|L_m^{-1}K_{mn}\|_F^2\), so this mainly helps DTC/FITC (no / cheaper trace). |
| **FITC / heteroscedastic** | Replace \(\sigma\) by \(\Lambda^{1/2}\), \(\Lambda=\mathrm{diag}(K_{nn}-Q_{nn})+\sigma^2 I\). Scale column \(i\) of \(A\) and entry \(i\) of \(y\) by \(\Lambda_{ii}^{-1/2}\). Use \(\tfrac12\sum\log\Lambda_{ii}\) instead of \(N\log\sigma\); drop VFE trace. |
| **Plain DTC** | VFE without trace correction. |
| **\(K_{mm}\)** | Still needs chol (or jittered chol). If \(K_{mm}=\Phi\Phi^\top\) (e.g. RFF), LQ \(\Phi\) there too. |

### 5. Relationship to CIP-0012 / flag semantics

| Layer | CIP-0012 StableFactor (today) | CIP-001B (this) |
|-------|-------------------------------|-----------------|
| \(K_{uu}\) | chol \(U\) / \(L_m\) | unchanged |
| Whitening | \(B=U^{-\top}K_{uf}\) | same (\(A\propto B\)) |
| \(I+\beta BB^\top\) | `syrk` + `potrf` | **LQ of \([I,\,A]\) (or augmented \(C\))** |
| Grads | Murray chol/`trsm`/`syrk` adjoints | + LQ adjoint from Seeger et al. |

**Proposed Accept choice:** introduce a third runtime mode `SparseLinalgLQ` (CLI `--sparse-linalg lq`), keep StableFactor as default until soak proves LQ better on CIP-001A stress + mpmath gates; then decide whether to promote LQ to default or merge into `stable`. Do **not** silently change StableFactor behaviour mid-soak.

### 6. Non-goals

- Replacing Legacy (`pdinv`) archaeology.
- Multi-output / non-spherical sparse in v1 (same spherical scope as CIP-0012).
- Dropping \(K_{mm}\) jitter entirely — LQ does not remove the need for a well-posed \(K_{mm}\) factor when \(Z\) collapses.
- Porting MXNet `gelqf` operator wholesale; wrap LAPACK `gelqf`/`orglq` (or QR of transpose) in GPc/CMatrix style.

## Implementation Plan

1. **Prototype (offline).** Dense NumPy/SciPy: LQ bound vs CIP-0012 Cholesky bound vs full \(O(N^3)\) VFE on small \(N\); confirm \(\sim 10^{-13}\) agreement on well-conditioned cells (already sketched).
2. **LAPACK wrapper.** `CMatrix` (or thin helper) for LQ/QR of wide matrices; diagonal-sign normalisation helper; log-abs-diag.
3. **Forward path.** DTCVAR (VFE) first: build \(A\), augmented \(C\), extract \(L,c,r\), evaluate \(\mathcal{F}\). Then DTC, then FITC (\(\Lambda\) scaling).
4. **Prediction** matching §3.
5. **LQ reverse-mode grads** for hyperparameter / \(Z\) / \(\beta\) (Seeger et al. Appendix); FD gate + dual-path vs StableFactor.
6. **CLI / enum** `--sparse-linalg lq`; document in [docs/numerics.md](../docs/numerics.md).
7. **Gates.** Extend `numerical_stability/` stress + `diagSparseStable` / mpmath: LQ vs StableFactor LL/grad; crowded-\(Z\) FD util; optional hard cell where Gram+chol degrades and LQ holds.

## Backward Compatibility

- Default remains CIP-0012 StableFactor until an explicit soak decision.
- `.model` files still omit sparse-linalg mode (runtime only), same as CIP-0012.
- MatIO fixtures unchanged; LL must match StableFactor within CIP-001A budgets on mild cells.

## Testing Strategy

- Unit: LQ of \([I,\,A]\) vs `chol(I+AA^\top)` (abs diag); \(r^2\) vs Woodbury; sign-flip helper.
- Dual-path: LQ vs StableFactor on `testCGp` / `diagDtcvar` toys (\(\lvert\Delta\mathrm{LL}\rvert < 1\mathrm{e}{-10}\) mild).
- Stress: CIP-001A crowded-\(Z\) and \(\gamma\times M\) slice — expect LQ FD / LL util not worse than StableFactor; document if still limited by \(K_{mm}\).
- Reference: mpmath recompute of bound from dumped \(K_{mm},K_{mn},y,\sigma\).
- CI: numerical-stability job + existing matio/CGp gates with `lq` optional matrix entry or env `GPC_SPARSE_LINALG=lq`.

## Implementation Status

- [ ] Proposed design reviewed (Accept)
- [ ] LAPACK LQ/QR wrapper + sign policy
- [ ] DTCVAR forward LL via augmented LQ
- [ ] DTC + FITC variants
- [ ] Prediction
- [ ] LQ adjoint gradients + FD gate
- [ ] CLI mode + docs
- [ ] Stress / mpmath soak; default decision

## References

- M. Seeger, A. Hetzel, Z. Dai, E. Meissner, N. D. Lawrence (2017). *Auto-differentiating linear algebra.* arXiv:1710.08717 — §3.2 sparse GP (Cholesky whitening); §3.3 BLR (LQ of \([I,\alpha^{1/2}X]\)); §4.3 LQ forward/backward.
- Titsias (2009). Variational learning of inducing variables in sparse GPs — collapsed bound \(\mathcal{F}\).
- [CIP-0012](cip0012.md) — whitened StableFactor + Murray/Seeger chol adjoints (prerequisite).
- [CIP-000E](cip000E.md) — original StableFactor flag.
- [CIP-0013](cip0013.md) / [CIP-001A](cip001A.md) — bench and stress evidence that Gram-side errors remain.
- Quiñonero-Candela & Rasmussen (2005) — DTC / FITC taxonomy (not to be confused with LAPACK QR).

## Author and Date

Neil Lawrence, 2026-11-18.
