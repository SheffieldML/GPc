---
author: "Neil Lawrence"
created: "2026-11-15"
id: "000D"
last_updated: "2026-11-16"
status: "Implemented"
compressed: false
related_requirements: ["0006"]
related_cips: ["000C"]
tags:
- cip
- kernels
- multioutput
- coregionalize
- gpy
title: "Multi-output covariance via Coregionalize"
---

# CIP-000D: Multi-output covariance via Coregionalize

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-0006](../requirements/req0006_multioutput-covariance.md). It is deliberately separate from [CIP-000C](./cip000C.md) (single-output kernel coverage).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Implement GPy’s multi-output kernels as GPy builds them, not as a new GP class.

GPy’s `GPCoregionalizedRegression` is a thin wrapper. The objects that define the covariance are in `GPy.util.multioutput`: `build_XY` for the stacked design, `ICM` for one spatial kernel times one `Coregionalize`, and `LCM` for a sum of those products. This CIP adds those three constructions on the existing `CKern` contracts (`CCoregionalizeKern`, `activeDims`, `CTensorKern`, `CCmpndKern`) and checks them against GPy. It does not port `MultioutputGP`, `IndependentOutputs`, `Hierarchical`, or derivative kernels.

The design below is locked. The earlier draft left stacking and the tensor-versus-façade choice open; both are now the GPy choices.

## Motivation

Correlated outputs need a cross-output factor \(B\). Importing more single-output kernels (CIP-000C) does not provide it. GPy’s usual model is the intrinsic coregionalization model

\[
k_{\mathrm{ICM}}((x,i),(x',j)) = k_{\mathrm{space}}(x,x')\, B_{ij},
\qquad
B = WW^\top + \mathrm{diag}(\kappa),
\]

and the linear coregionalization model, which is a sum of such terms. Users who already write `ICM` / `LCM` in GPy should get the same matrices from GPc.

`CTensorKern` today multiplies its children on the full input matrix. An RBF inside that product would treat GPy’s output-index column as a real input. Matching GPy requires each child to see only its active columns. That is `active_dims`, not a special case inside `CGp`.

## Detailed Description

### What GPy actually builds

From `GPy/util/multioutput.py` and `GPy/kern/src/coregionalize.py` (GPy devel, the same objects as the 1.x `GPCoregionalizedRegression` path):

1. **`build_XY(X_list, Y_list)`.** One array per output. Row counts may differ; input dimension \(q\) may not. Stack vertically. Append one column for the output index \(0 \ldots D-1\) in list order, as the **last** column. In NumPy that column is still a float array; `Coregionalize` casts with `np.asarray(X, dtype=int)` at use time. \(Y\) is the same vertical stack. The index is not interleaved.

2. **`Coregionalize`.** \(B = WW^\top + \mathrm{diag}(\kappa)\), cached when parameters change. \(W\) is \(D \times R\), unconstrained. \(\kappa\) is length \(D\), constrained positive with GPy’s `Logexp`. The kernel’s own input is the index column (`input_dim=1`). \(K_{nm} = B_{i_n, i_m}\). Input gradients are identically zero. \(R > D\) is legal; GPy warns and continues. Defaults, when the caller does not pass values: \(W_{dr} = 0.5\, \mathcal{N}(0,1) / \sqrt{R}\), \(\kappa_d = 0.5\). Tests pass \(W\) and \(\kappa\) in, so they do not depend on that draw.

3. **`ICM(input_dim, num_outputs, kernel, W_rank)`.** `kernel.prod(Coregionalize(1, num_outputs, active_dims=[input_dim], rank=W_rank))`. The spatial kernel sees columns \(0 \ldots q-1\). `Coregionalize` sees column \(q\). The product is elementwise, which is GPy’s `prod` and the `**` operator. White noise is not a factor inside this product (`CTensorKern::addKern` already rejects white).

4. **`LCM`.** Sum of ICMs, one per spatial kernel, each with its own \(B\). That sum is GPy’s `Add`, which is `CCmpndKern`.

5. **Noise.** `GPCoregionalizedRegression` uses `MixedNoise`: one Gaussian variance per output, selected by the index column. The marginal covariance is \(K_{\mathrm{ICM}} + \mathrm{diag}(\sigma^2_{i_n})\). A single shared white kernel is a different model. This CIP adds that diagonal as its own kernel, summed **outside** the product, so the comparison with GPy’s marginal covariance does not require a new likelihood class.

`Private` in GPy is an ICM with \(W\) fixed at 0 and all but one \(\kappa\) fixed at 0. It is not a new class here. Fixing parameters can come later.

### Locked GPc design

| GPy object | GPc object | Stream type |
|---|---|---|
| `Coregionalize` | `CCoregionalizeKern` | `coregionalize` |
| `Prod` / `**` with `active_dims` | `CTensorKern` plus per-child `activeDims` | `tensor` (existing) |
| `Add` of ICMs | `CCmpndKern` | `cmpnd` (existing) |
| `ICM(...)` | factory helper that builds the tensor and sets active columns | — |
| `LCM(...)` | factory helper that sums those tensors | — |
| `MixedNoise` of Gaussians | `CIndexWhiteKern`, added beside the ICM | `indexwhite` |

**Index column storage (no typed matrix).** `CMatrix` remains all `double`. There is no parallel integer matrix and no change to the BLAS layout. GPy stores the same column as float and casts in `Coregionalize`; GPc does the same. Spatial columns stay ordinary doubles. The last column holds \(0, 1, \ldots, D-1\) as doubles. Only `CCoregionalizeKern` and `CIndexWhiteKern` interpret that column as an index. `activeDims` keeps the RBF (and any other spatial factor) from reading it, so a lengthscale never sees “output 2” as a continuous coordinate.

**Index cast rule** (shared by `CCoregionalizeKern` and `CIndexWhiteKern`):

1. \(i \leftarrow \mathrm{llround}(x)\) (nearest integer; exact integers and values within a rounding ulp of an integer both work).
2. `BOUNDCHECK` that \(|x - i| < \varepsilon\) with \(\varepsilon =\) `ndlutil::MATCHTOL` (or a named constant next to it). Reject `1.7`; do not silently truncate as NumPy `dtype=int` would.
3. `BOUNDCHECK` that \(i \in \{0,\ldots,D-1\}\).

Fixtures and demos write exact integer values (`0.0`, `1.0`, …). Loaders (SVM-light and friends) write the index as an ordinary numeric feature; they do not need a new type.

**`activeDims`.** `CKern` stores the columns it reads. The default is \(0 \ldots \mathrm{inputDim}-1\) with `X.ncols == inputDim`, and that path stays the current element code, so single-output models and the matio suite do not change. A non-default list means `X` may be wider than `inputDim`. Stationary `computeElement` / `dist2` calls use those columns and do not copy a submatrix on each entry. A kernel that does not honour `activeDims` cannot be placed in an ICM; constructing that product throws rather than silently folding the index into the lengthscale. `CCoregionalizeKern` always has `activeDims = [q]` and `inputDim = 1`. Its `getGradX` is zero on every column, including the index.

**`CCoregionalizeKern`.** Parameters, in GPy’s link order: \(W\) row-major (\(D R\) values, no transform), then \(\kappa\) (\(D\) values, `CExpTransform`, the existing real-to-positive map). Stream fields: `outputDim`, `rank`, `W`, `kappa`. Rebuild \(B\) when those parameters change. \(B_{ij}\) is then \(O(1)\). Index values follow the cast rule above. Shapes of \(W\) and \(\kappa\) are checked; nothing is truncated. \(R > D\) is allowed and warned, as in GPy.

**`CIndexWhiteKern`.** `inputDim = 1`, `activeDims = [q]`, \(D\) positive variances. Diagonal entry \(\sigma^2_{i_n}\), off-diagonal 0, input gradient 0. Same index cast rule as coregionalize. It is added with `CCmpndKern`, never passed to `CTensorKern::addKern`. This is the marginal-covariance form of GPy’s per-output Gaussian noise. The existing single `CWhiteKern` remains the shared-noise model and is not the GPy default.

**`CGp`.** No new approximation and no new multi-output class. The GP sees the stacked \(X\) (\(N \times (q+1)\)) and stacked \(y\) (\(N \times 1\)). Prediction for output \(d\) uses rows whose last column is \(d\).

**CLI.** One documented construction for a two-output toy: stacked SVM-light (or equivalent) file with the index column, kernel `tensor` of `rbf` and `coregionalize`, plus `indexwhite`. Defaults of `gp` stay single-output. Exact flag letters can wait until implementation; the kernel types and column layout cannot.

### Explicit non-goals

- `IndependentOutputs`, `Hierarchical`, `MultioutputKern`, `MultioutputGP`, `MultioutputDerivativeKern`, `DiffKern`
- Heterogeneous non-Gaussian likelihoods per output
- SDE multi-output and ODE/LFM families
- Changing default single-output `gp` behaviour
- The CIP-000C kernel list

### Alternatives considered

- **A private ICM class that slices \(X\) and never adds `activeDims`.** Same \(K\) for this one product, and a second composition graph the next kernel would have to know about. Rejected: GPy’s mechanism is `active_dims` on the children of `Prod`.
- **Interleaved rows, or the index in the first column.** Rejected: `build_XY` appends the index and stacks blocks.
- **A typed / heterogeneous `X` (int index column, double spatial columns).** Rejected: `CMatrix` and BLAS are double throughout; GPy itself stores the index as float and casts at use. Casting in the two index-reading kernels is enough.
- **Silent NumPy-style truncation (`dtype=int`).** Rejected for GPc: require near-integer plus range checks so a mistyped `1.7` fails loudly.
- **Shared white noise as the GPy comparison.** Rejected for the likelihood check. It remains available as today’s `CWhiteKern`.
- **Full wrapper port.** Still rejected. `GPCoregionalizedRegression` adds nothing to the covariance beyond `ICM` and `MixedNoise`.

### Dependency on CIP-000C

None for the coregionalize factor. The reference spatial kernel is the existing `CRbfKern`. Further spatial kernels participate in LCM once they read `activeDims`.

## Implementation Plan

1. **`activeDims` on `CKern`.** Default full-width path unchanged. Active-column distance without a per-entry copy. Stationary kernels used as ICM/LCM factors call it.
2. **`CCoregionalizeKern`.** Compute, diagonal, parameter gradients, zero `getGradX`, stream fields above, factory type `coregionalize`.
3. **Helpers `icm` and `lcm`.** Set active columns as GPy’s functions do. `lcm` returns a `CCmpndKern`.
4. **`CIndexWhiteKern`.** Factory type `indexwhite`. Summed outside the tensor.
5. **Fixtures.** Generate \(K\), \(\partial K / \partial W\), \(\partial K / \partial \kappa\), and the index-white diagonal from GPy `ICM` and `LCM` on a stacked two-output design, including unequal \(N_d\) and a near-duplicate \(x\). `testKern` loads them.
6. **Demo.** Two-output toy, stacked layout, `rbf` ICM plus `indexwhite`. Document the layout next to the example. One log-likelihood comparison against `GPCoregionalizedRegression` at fixed parameters (same \(W\), \(\kappa\), lengthscale, variance, per-output noise).

## Backward Compatibility

- New kernel types only. Existing streams and single-output `gp` are unchanged.
- Default `activeDims` keeps current `computeElement` behaviour when `X.ncols == inputDim`.
- `CTensorKern` still rejects a white component inside the product.

## Testing Strategy

- `coregionalize` alone, against GPy, on exact integer values stored as doubles, including an out-of-range index and a non-integer (`1.7`) that must throw.
- ICM: `rbf` with `activeDims = [0]` times `coregionalize` with `activeDims = [1]`, against `GPy.util.multioutput.ICM`, for \(K\) and for gradients of \(W\), \(\kappa\), and the RBF parameters. The RBF lengthscale must not respond to the index column (`getGradX` on that column is zero from both factors).
- LCM: two ICMs summed, against `LCM`, so the two \(B\) matrices are not tied.
- `indexwhite` diagonal against a GPy `MixedNoise` marginal (or the equivalent \(\mathrm{diag}(\sigma^2_{i_n})\)).
- Stream round-trip of `coregionalize` and `indexwhite`.
- Existing matio kernel tests stay green, which is the check that default `activeDims` did not move single-output results.
- Demo smoke plus one fixed-parameter log-likelihood comparison. Training to convergence is not the gate.

## Related Requirements

- [REQ-0006](../requirements/req0006_multioutput-covariance.md) — the outcome this CIP implements.

## Related

- [CIP-000C](cip000C.md) — single-output GPy kernels. Not a blocker. Those kernels become LCM factors when they honour `activeDims`.

## Implementation Status

- [x] Design locked to GPy `build_XY`, `Coregionalize`, `ICM`, `LCM`, and per-output Gaussian noise
- [x] Index column stored as `double`; cast rule (`llround` + near-integer + range) locked
- [x] `activeDims` with an unchanged default path (`CRbfKern` honours it)
- [x] `CCoregionalizeKern` + factory (uses cast rule)
- [x] `icm` / `lcm` helpers (`createIcmKern`, `createLcmKern`)
- [x] `CIndexWhiteKern` + factory (uses cast rule)
- [x] `testCoregionalize` (index cast, B, ICM product, index-white, streams, tensor rejection)
- [x] Layout note: `docs/coregionalize.md`; fixture generator: `scripts/gen_cip000D_coreg_fixtures.py`
- [ ] Optional: wire generated `.mat` into `testKern`; fixed-parameter LL vs `GPCoregionalizedRegression`

## References

- GPy `GPy.kern.src.coregionalize.Coregionalize`
- GPy `GPy.util.multioutput.build_XY`, `ICM`, `LCM`, `build_likelihood`
- GPy `GPy.models.gp_coregionalized_regression.GPCoregionalizedRegression`
- GPy `GPy.examples.regression.coregionalization_toy` (list-of-outputs API; stacking is `build_XY`)
- Álvarez, Rosasco, and Lawrence, kernels for vector-valued functions (ICM/LCM background)
- Tenets: Compose Through Interfaces; Assert Invariants; Flexibility Without a Performance Tax; Ecosystem Interoperability
