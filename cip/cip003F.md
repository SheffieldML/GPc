---
author: "Neil Lawrence"
created: "2026-11-27"
id: "003F"
last_updated: "2026-11-27"
status: "Closed"
compressed: false
related_requirements: ["001A", "0015", "0002", "0016"]
related_cips: ["0035", "002A", "002D", "0028", "003C", "003D", "003E", "0041"]
tags:
- cip
- performance
- deepgp
- svgp
- kernels
title: "Deep GP training speed (batched gradients, O(B) SVGP steps)"
---

# CIP-003F: Deep GP training speed (batched gradients, O(B) SVGP steps)

> CIPs describe HOW to achieve requirements (WHAT). This CIP serves [REQ-001A](../requirements/req001A_deep-gaussian-processes.md) (deep GPs) and [REQ-0015](../requirements/req0015_stochastic-variational-sparse-gp.md) (SVGP), with gates under [REQ-0002](../requirements/req0002_runnable-verification.md). GPU work stays under [REQ-0016](../requirements/req0016_gpu-training-decision.md) and is out of scope. The per-layer `DeepGP` façade work originally drafted here is now [CIP-0041](./cip0041.md).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [x] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Running the deep GP on MNIST (60 000 × 784, ten outputs, 3 layers, \(M=100\), hidden width 20, batch 256, CPU) gave about 180 ms per Adam step with the RBF covariance and about 470 ms with the MLP covariance, against about 5 ms for a JAX MLP of comparable arithmetic per example. Profiling shows the gap is GPc implementation, not algorithm: per-element loops in kernel input gradients and in the variational data term, and a minibatch path in `CSvgp` whose cost still grows with \(N\). This CIP replaces those loops with BLAS-3 products and makes every minibatch step \(O(B)\). Results are unchanged up to round-off; every change is gated by an equivalence test against the code it replaces.

## Motivation

### Where a step goes

Profile of 60 Adam steps, `DeepGP(L=3, kernel='rbf', inducing='points', M=100, hidden=20)` on full MNIST, with lazy full-data matrices (inclusive shares in brackets):

| Share (self) | Site | What it does |
|---|---|---|
| 48% | `CRbfKern::getGradX` (via `CInducingPoints::gradFromKuuKuf`, 58% inclusive) | \(\partial K_{fu}/\partial \mathbf{x}\) for the 784-D first layer, one datum at a time, recomputing distances and `exp` already in `Kuf` |
| 10% | `CSvgp::dataTermQGradsAt` | \(\partial/\partial\tilde{\mathbf m}\), \(\partial/\partial\tilde S_j\) accumulated with `getVal`/`setVal` over \(B\times M^2\) per output |
| 6% | `CInducingPoints::gradFromKuuKuf` (own code) | element loops around the kernel calls |
| 20% | all BLAS/LAPACK (`symm`, `gemm`, `trsm`, `syrk`) | the work that should dominate |

Achieved throughput is about 5–6 GFLOP/s (≈1 GFLOP of arithmetic per step), against about 110 GFLOP/s for the JAX network on the same two cores. A step does about the same arithmetic per training example as the network; the per-epoch gap (≈19×) is implementation.

### The MLP covariance is slower again

`CMlpKern` has a GEMM `compute` (CIP-002D QW5) but no batched input or inducing gradient: `getGradX` recomputes `dotRowRow` and norms per pair and there is no `inducingGradZ` override, so \(\partial/\partial Z\) falls back to one `getGradX` per inducing point. Weight-space ridges (`CMlpKern::wsRidgeCrossCompute`) fill \(K_{fu}\) element by element through `mlpRidgeKfuElement`, recomputing per-window constants for every datum. Measured: about 470 ms per step with inducing points and about 590 ms with weight-space windows on the MNIST configuration, against 180 ms for RBF.

### The plain SVGP path is still \(O(N)\) per step

The lazy full-data change (257f1db, backlog `2026-11-26_svgp-lazy-full-data`) removed the \(O(N)\) work from deep-GP layer cores, which use the explicit-input API. `CSvgp::elboGrad(batch)` without input gradients still scatters site adjoints into a full \(M\times N\) `Abar` and calls `trsmAdjointUT` against the full `A`, and the q-gradient and natural-gradient paths read `A` directly for large batches. A one-layer `DeepGP` (which delegates to `CSvgp::objectiveGradBatch`) and `SparseGPRegression` in stochastic mode therefore pay \(O(NMd)\) per step; on MNIST the one-layer run spent most of each step rebuilding the full-data matrices (for comparison, the eager two-layer step was 880 ms). A local patch routing small batches (\(4B<N\)) through `elboGradAt` and restricting the full-data guards to the full ordered batch gave 47 ms with `testSvgp`, `testDeepGp` and `testSviBayesGplvm` unchanged.

## Detailed Description

**A1. \(O(B)\) minibatch steps for `CSvgp`.**
- `elboGrad(batch, gX=0)`: for \(4B<N\), call `elboGradAt` on the gathered batch inputs (as `formAB` already does for marginals, CIP-002D QW6). The full-data path remains for large or full batches.
- `dataTermQGrads(batch)` and `naturalGradientStep(gamma, batch)`: same routing, through the explicit-input variants.
- Lazy full-data matrices become the default for stochastic use: `ensureFullData()` is called only where `A`, `diagK` or `diagQ` are read on the full ordered batch or large batches. `setLazyFullData(false)` keeps the eager behaviour for callers that want it.

**A2. Batched input and inducing gradients in kernels.** Add a hook beside the existing `inducingGradZ`,

```cpp
// gX (B x d) += sum_m gKuf(m, b) dK(z_m, x_b)/dx_b, using cached K_uf when given.
virtual void inducingGradX(CMatrix& gX, const CMatrix& Z, const CMatrix& X,
                           const CMatrix& gKuf, const CMatrix* KufCached = 0) const;
```

with the current per-datum `getGradX` loop as the default, and use it in `CInducingPoints::gradFromKuuKuf` when `wantX`. Overrides, with \(\mathbf G\) the \(B\times M\) upstream gradient (transpose of `gKuf`):

- **RBF / RBF-ARD.** With \(\mathbf W = \mathbf G\circ\mathbf K_{fu}\) and inverse lengthscales \(\bm\Lambda\),
  \(\partial = \bm\Lambda\left(\mathbf W\mathbf Z - \operatorname{diag}(\mathbf W\mathbf 1)\,\mathbf X\right)\): one elementwise product and one GEMM, reusing the cached kernel.
- **Linear.** \(\partial = \mathbf G\mathbf Z\bm\Sigma\).
- **MLP / MLP-ARD.** GPc's convention is \(k = \sigma^2\arcsin(u)\), \(u = n/\sqrt{ab}\), with \(n = w\,\mathbf x^\top\mathbf z + \beta\), \(a = w\|\mathbf x\|^2+\beta+1\), \(b = w\|\mathbf z\|^2+\beta+1\). Then \(\partial k/\partial\mathbf x = c\left(\mathbf z - \frac{n}{a}\mathbf x\right)\) with \(c = \sigma^2 w/\left(\sqrt{1-u^2}\sqrt{ab}\right)\). With \(\mathbf C\) and \(\mathbf N\) the \(B\times M\) matrices of \(c\) and \(n\) (both available from the GEMM in `compute`),
  \(\partial = (\mathbf G\circ\mathbf C)\,\mathbf Z - \operatorname{diag}\!\left(\frac{(\mathbf G\circ\mathbf C\circ\mathbf N)\mathbf 1}{\mathbf a}\right)\mathbf X\),
  and symmetrically for \(\partial/\partial\mathbf Z\), which becomes the missing `CMlpKern::inducingGradZ` override. ARD applies the same formulae to \(\mathbf X\bm\Sigma^{1/2}\) and chains.

**A3. Weight-space ridges in bulk.** In `wsRidgeCrossCompute` and the ridge gradient functions (`wsRidgeGradParams`, `GradX`, `GradVA`, `GradS`), form \(\hat{\mathbf X}\mathbf V^\top\) with one GEMM, compute per-window constants (\(A_m\), \(\mu_m\), \(c_s\)) once, and apply the error function and its derivative elementwise. This is the "MLP bulk path" left open in CIP-002D.

**A4. Variational data term as BLAS-3.** In `dataTermQGradsAt` (and the full-batch `dataTermQGrads`), accumulate \(\mathbf g_{\tilde m} \mathrel{+}= \mathbf A_B(c\,\mathbf g^\mu)\) and \(\mathbf g_{S_j} \mathrel{+}= \mathbf A_B\operatorname{diag}(c\,\mathbf g^v_j)\mathbf A_B^\top\) with one GEMM per output, as `marginalsGradAt` already does. In `elboGradAt`, form the site adjoint \(\bar{\mathbf A}\) as \(\tilde{\mathbf m}_j(c\,\mathbf g^\mu_j)^\top + 2(\tilde S_j\mathbf A_B - \mathbf A_B)\operatorname{diag}(c\,\mathbf g^v_j)\) with a column scaling and a GEMM instead of the triple loop.

**A5. KL gradient from the stored Cholesky factor.** Fold in backlog `2026-11-22_csvgp-kl-grad-from-chol`: in Cholesky mode, \(\partial\mathrm{KL}/\partial L = L - \operatorname{diag}(1/L_{ii})\) on the lower triangle and \(\partial\mathrm{KL}/\partial\tilde{\mathbf m} = \tilde{\mathbf m}\), removing the per-output `pdinv` (\(D_\ell M^3\) per step).

**A6. Timing cases.** Add to `bench/timing_cases.json` a synthetic deep GP case that needs no download: \(N\in\{6000, 60000\}\) inputs of dimension 784 drawn uniformly on \([0,1]\), ten one-hot outputs, batch 256, \(M=100\), hidden width 20, \(L\in\{1,3\}\), kernels RBF and MLP, inducing points and (MLP) weight-space windows. Record milliseconds per step and the lscpu model. Gates are relative, measured on the same machine at the start and end of the CIP (CI runner variance made absolute targets unreliable before).

### Out of scope

- Per-layer façade options, MLP initialisation, `mlpard`, prior-draw Jacobian and BLAS thread control ([CIP-0041](./cip0041.md)).
- Multi-class likelihoods ([CIP-003E](./cip003E.md)); the MNIST timing case uses Gaussian one-hot targets.
- GPU backends and single precision ([REQ-0016](../requirements/req0016_gpu-training-decision.md)).
- Convolutional covariance functions.
- Fixes to the `deepgp-mlp` notebooks themselves (Student-t interval computation, banknote fetch); they belong to that repository.

### Expected effect (estimates only, not commitments)

Measured on the development machine (macOS, `RELEASE=1`, `OPENBLAS_NUM_THREADS=4`), synthetic MNIST-shaped case: \(q=784\), 10 one-hot outputs, \(L=3\), hidden width 20, \(M=100\), batch 256, median of 9 steps, three repeats. The baseline is the commit before Phase 1, built with the same benchmark. Timings vary by about 10% between runs on this machine; treat the ratios, not the digits, as the result. The same cases are in `bench/timing_cases.json` (`deepgp_*`) for the reference machine.

| Configuration | \(N\) | Before (ms/step) | After (ms/step) |
|---|---|---|---|
| RBF, inducing points | 6000 | 112–123 | 45–48 |
| RBF, inducing points | 60000 | 111–117 | 50–68 |
| MLP, inducing points | 6000 | 320–327 | 77–78 |
| MLP, inducing points | 60000 | 328–384 | 82–83 |
| MLP, weight-space windows | 6000 | 592 | 80–92 |
| MLP, weight-space windows | 60000 | 592–596 | 86–101 |

The \(N\) dependence that the plain SVGP path showed (full-data rebuild every step) is removed by the \(O(B)\) routing; the remaining difference between \(N=6000\) and \(N=60000\) is cache and gather cost. The single-layer SVGP path is measured by `benchGPc --kind svgp`.

Two notes on what was implemented against what was proposed. First, A5 (KL gradient from the Cholesky factor) was already satisfied: `klGradPackedQ` forms it from the factor, so the backlog item was stale and no code changed; Phase 3 consists of the BLAS-3 data-term gradients. Second, the equivalence tests for the SVGP routing and the data-term gradients use a tolerance of \(10^{-10}\) (absolute, on values of order one) rather than the \(10^{-12}\) stated in the gates; the kernel-level gradient and ridge tests meet \(10^{-12}\).

## Implementation Plan

1. **Phase 0 — baseline.** A6 timing cases; record numbers on the reference machine with fixed threads.
2. **Phase 1 — A1.** \(O(B)\) routing in `elboGrad`, `dataTermQGrads`, `naturalGradientStep`; lazy default.
   - Gates: small-batch routed results equal the full-data path on the same batch to \(10^{-10}\) (ELBO, all gradients, natural-gradient step); existing `lazyEquivalenceCase`; step time at \(N=60000\) within 1.2× of \(N=6000\).
3. **Phase 2 — A2.** `inducingGradX` hook; RBF, RBF-ARD, linear, MLP, MLP-ARD overrides; `CMlpKern::inducingGradZ`.
   - Gates: batched equals per-element `getGradX` to \(10^{-12}\) relative for every kernel and both \(\mathbf X\) and \(\mathbf Z\); finite differences through `CDeepGp` unchanged.
4. **Phase 3 — A4, A5.** GEMM data term and site adjoint; KL from the Cholesky factor.
   - Gates: q-gradients equal the loop versions to \(10^{-12}\); finite differences on packed \(q\) parameters (`testSvgp`); no `pdinv` in Cholesky mode.
5. **Phase 4 — A3.** Bulk weight-space ridges.
   - Gates: bulk equals `mlpRidgeKfuElement` and the element gradients to \(10^{-12}\); `testInducing` and CIP-0028 variance-captured gates unchanged.
5. **Phase 5 — re-measure.** Rerun A6 and the MNIST deep GP, replace the estimate table above; record before/after in this CIP and in `docs/deepgp.md`.

## Backward Compatibility

- Numerical results are unchanged by A1–A5 up to round-off (all gates are equivalence tests against the current code paths).
- Lazy full-data matrices become the default; `setLazyFullData(false)` restores eager construction.
- New kernel hook has a default implementation, so third-party kernels keep working.

## Testing Strategy

- Equivalence tests for every batched path against the per-element code it replaces (A2–A4), on random inputs and on the CIP-0035 composite task, in the hard tier.
- Finite-difference checks through `CSvgp` and `CDeepGp` for each kernel and inducing family, in the hard tier.
- Routing tests for A1 on small, half and full batches, including the \(N/B\) scaling.
- Timing cases (A6) in the numerics/bench tier with relative gates; not in the hard tier.

## Related Requirements

- [REQ-001A](../requirements/req001A_deep-gaussian-processes.md) — deep GPs that train on minibatches at practical cost.
- [REQ-0015](../requirements/req0015_stochastic-variational-sparse-gp.md) — SVGP minibatch cost independent of \(N\).
- [REQ-0002](../requirements/req0002_runnable-verification.md) — runnable verification for each change.
- [REQ-0016](../requirements/req0016_gpu-training-decision.md) — GPU remains a separate decision.

## Implementation Status

- [x] CIP accepted
- [x] Phase 0: timing baseline
- [x] Phase 1: \(O(B)\) SVGP steps
- [x] Phase 2: batched input and inducing gradients
- [x] Phase 3: GEMM data term (KL from Cholesky already in place)
- [x] Phase 4: bulk weight-space ridges
- [x] Phase 5: re-measure and document ([docs/deepgp.md](../docs/deepgp.md))

## References

- [CIP-0035](./cip0035.md) — deep GP design and prototype; [CIP-002D](./cip002D.md) — CPU quick wins and the open MLP bulk path; [CIP-003C](./cip003C.md) — earlier façade gaps; [CIP-0041](./cip0041.md) — per-layer façade; [CIP-003E](./cip003E.md) — multi-class likelihoods.
- Backlog: `2026-11-26_svgp-lazy-full-data` (Completed, timing not yet reproduced), `2026-11-22_csvgp-kl-grad-from-chol` (folded into A5).
