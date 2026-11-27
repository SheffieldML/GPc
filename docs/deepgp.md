# Deep Gaussian processes: model, training step, and cost

Reference for `CDeepGp` and the machinery under it. For command-line use see the [tutorial](tutorials/deepgp.md); for the design history see [CIP-0035](../cip/cip0035.md) (model), [CIP-003F](../cip/cip003F.md) (training speed) and [CIP-0041](../cip/cip0041.md) (per-layer façade, proposed). Requirement: [REQ-001A](../requirements/req001A_deep-gaussian-processes.md).

## Model

`CDeepGp` stacks \(L\) sparse variational GP layers (`CSvgp`) trained by doubly stochastic variational inference. Layer \(\ell\) maps \(D_{\ell-1}\) inputs to \(D_\ell\) outputs. Each layer has

- a fixed linear mean map \(P_\ell\) (so the layer is a residual correction to a linear map),
- one kernel (`CKern`), borrowed from the caller,
- an inducing family (`CInducing*`): `points`, `weightspace` (CIP-0028 ridge windows) or `hermite` (CIP-003D),
- a whitened variational posterior \(q(v)=\mathcal N(m,S)\) over \(M_\ell\) inducing values.

With whitening, \(A=L^{-1}K_{uf}\) where \(K_{uu}=LL^\top\). The predictive marginal of a layer at input \(h\) is \(m(h)=P^\top h + A^\top m\) with variance \(k(h,h)-a^\top a + a^\top S a\). The \(-a^\top a\) term is the Nyström residual; `setDropResidual(true)` removes it, giving a bound on the finite-network model instead.

Hidden layers are sampled by reparameterisation (`setNumSamples(K)` draws per datum). The final layer uses `elboGradAt` and may carry a non-Gaussian likelihood (`setLikelihood`, `adoptLikelihood`). The ELBO is

\[
\mathcal L = \frac{N}{B}\sum_{n\in\text{batch}} \mathbb E_q[\log p(y_n\mid f_n)] - \sum_{\ell}\mathrm{KL}\big(q(v_\ell)\,\|\,\mathcal N(0,I)\big).
\]

Training uses minibatch Adam through `CStochasticOptimisable`: `objectiveGradBatch(batch, g)` returns the negative ELBO estimate and its gradient with respect to all layers' parameters (kernel, inducing, \(m\), Cholesky factor of \(S\), noise).

## Anatomy of one step

For a batch of \(B\) points, \(K\) samples, \(L\) layers and \(M\) inducing points per layer:

1. **Forward.** Push the batch through the layers. Each hidden layer evaluates \(K_{uf}\) for the batch (\(M\times B\)), forms \(A=L^{-1}K_{uf}\) with a triangular solve, and draws a sample. Cost \(O(M^2 B)\) plus the kernel evaluation.
2. **Final layer.** `CSvgp::elboGradAt` evaluates the data term and the adjoint with respect to \(A\), \(m\), \(S\), the kernel parameters, the inducing parameters and the layer input.
3. **Backward.** Hidden layers receive the input adjoint from the layer above and run `marginalsGradAt`. Each produces gradients for its kernel, inducing set, \(m\), \(L_S\) and its own layer input.
4. **KL.** The KL gradient is formed analytically from the Cholesky factor of \(S\) (`klGradPackedQ`); it does not depend on \(N\) or \(B\).

All of the above is \(O(B)\) in the data size: nothing is rebuilt over all \(N\) points during a step. (This was not true of the single-layer SVGP path before CIP-003F; see below.)

## What makes a step cheap

CIP-003F removed the avoidable costs. They are described here because they constrain how new kernels and inducing families should be written.

| Mechanism | Where | Rule for new code |
|---|---|---|
| \(O(B)\) routing | `CSvgp::elboGrad(batch)` calls `elboGradAt` on the gathered rows when \(4B<N\). Full-data matrices (`ensureFullData`) are built lazily and are not built for small batches. | Never touch full-data state in a batch step. `setLegacyPaths(true)` restores the old path for testing. |
| BLAS-3 data-term gradients | `accumulateDataTermQGrads`: \(g_m\) by gemv, each \(g_{S_j}=A\,\mathrm{diag}(c\,g_\mathrm{var})A^\top\) by gemm. | Express gradients as matrix products, not per-datum loops. |
| Batched `inducingGradX` | `CKern::inducingGradX(gX, Z, X, gKuf, KufCached)` accumulates \(\sum_m g_{K_{uf}}(m,b)\,\partial k(z_m,x_b)/\partial x_b\) in one pass. RBF, RBF-ARD, linear, MLP, MLP-ARD and compound kernels override it; the base class loops `getGradX` and is the test oracle. | A new kernel works with the default but should override with a GEMM form before it is used in a deep stack. |
| `CMlpKern::inducingGradZ` | Inducing-point gradient for MLP and MLP-ARD (Kuf term, symmetric Kuu term, diagonal term). | |
| Bulk ridge windows | `RidgeBulk` in `CKern.cpp`: one GEMM for \(\hat X V^\top\), per-pair scalars once, and \(PV\), \(P^\top\hat X\) by GEMM in the gradients. Shared by `CMlpKern`, `CMlpardKern` and (order 1) `CArcCosineKern`. | The per-element functions remain as the reference; `CKern::setLegacyRidge(true)` selects them. |

## Cost

Measured on the development machine (macOS, `RELEASE=1`, `OPENBLAS_NUM_THREADS=4`), \(L=3\), \(q=784\), 10 one-hot outputs, hidden width 20, \(M=100\), batch 256. Median of 9 steps; run-to-run variation is about 10%, so compare ratios.

| Configuration | \(N\) | Before CIP-003F (ms/step) | After (ms/step) |
|---|---|---|---|
| RBF, inducing points | 6000 | 112–123 | 45–48 |
| RBF, inducing points | 60000 | 111–117 | 50–68 |
| MLP, inducing points | 6000 | 320–327 | 77–78 |
| MLP, inducing points | 60000 | 328–384 | 82–83 |
| MLP, weight-space windows | 6000 | 592 | 80–92 |
| MLP, weight-space windows | 60000 | 592–596 | 86–101 |

Reproduce with

```sh
make RELEASE=1 benchGPc
OPENBLAS_NUM_THREADS=4 ./benchGPc --kind deepgp --kern mlp --inducing weightspace \
    --N 60000 --q 784 --M 100 --layers 3 --hidden 20 --batch 256 --reps 9
python3 bench/run_timing_cases.py          # the deepgp_* cases in bench/timing_cases.json
```

`run_timing_cases.py` fixes `OPENBLAS_NUM_THREADS=1`, so its numbers are single-thread and not comparable to the table.

The step cost is dominated by \(M^2B\)-scale BLAS work per layer and, for the MLP kernels, by kernel and gradient evaluation. The weight-space ridge path costs about the same as the point path once its work is in GEMMs.

## Verification

Changes to a kernel, inducing family or the SVGP core owe an equivalence test and a gradient check ([project tenet](../tenets/gpc/assert-and-verify.md)). The existing gates are:

| Test | What it checks |
|---|---|
| `testSvgp` | Routed and BLAS-3 paths equal the legacy paths (value, kernel, \(Z\), \(\beta\) and \(q\) gradients) over small, half and shuffled-full batches, for points and weight-space inducing. Tolerance \(10^{-10}\). Also checks that full-data state stays unbuilt for small batches. |
| `testInducingGradZ` | Batched `inducingGradX` against `CKern::inducingGradX` (\(10^{-12}\)) for each overriding kernel, with and without a cached \(K_{uf}\); `inducingGradZ` for MLP; bulk ridge \(K_{fu}\) and all five gradients against the per-element reference (\(10^{-12}\)). |
| `testDeepGp` | Whole-model ELBO against an exact reference in reducing limits, finite-difference gradient checks, per-layer inducing types, learned window widths, and prediction. |
| `testInducingHermite`, `testSvgpBlock`, `testSviBayesGplvm`, `testCGp` | Neighbouring users of the same kernels and SVGP core. |

`testKern` needs matio and does not build without it.

## Limits

- Hidden-layer kernels must be one of the supported `CKern` types with an `inducingGradX` override to reach the speeds above; others work through the slower default.
- Mean maps \(P_\ell\) are fixed at construction.
- Convolutional kernels (CIP-0040) and a per-layer Python façade (CIP-0041) are proposed, not implemented.
