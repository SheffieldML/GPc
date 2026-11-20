# Manifold Relevance Determination (two-view smoke)

**GPy counterpart:** `GPy.models.MRD`  
**Executable:** `demo_mrd` (library API; no dedicated `./mrd` CLI yet)  
**Design:** [CIP-0018](../../cip/cip0018.md) · [REQ-000C](../../requirements/req000C_manifold-relevance-determination.md)

Single-view Bayesian sibling: [bgplvm_oil100](./bgplvm_oil100.md).

## Build and run

```sh
make demo_mrd
./demo_mrd
```

The demo synthesises two small aligned views (\(N=12\)), builds a `CMrd` with per-view RBF-ARD kernels, runs a short optimisation, prints the ELBO / parameters, then calls `factorizeSpace` to label latent dims as shared vs private given a scale threshold.

## What to expect

- One shared \(q(\mathbf{X})\) and inducing set \(Z\) across views.
- Objective \(\sum_v \mathcal{L}_{\mathrm{VarDTC}}^{(v)} - \mathrm{KL}(q\|p)\) with the KL counted once.
- Soft private/shared structure from comparing per-view ARD input scales (GPy `factorize_space` spirit).

`./testMrd` covers ELBO consistency, grads, stream, factorise helpers, and a committed GPy ELBO fixture (`testdata/cip0018/mrd_rbf_tiny.txt`, regenerated with `scripts/gen_cip0018_mrd_fixtures.py`). The fixture uses a single-KL assembly of two GPy `BayesianGPLVM` views (same path as `CMrd`); raw `GPy.models.MRD` MiniBatch VarDTC drifts slightly and is not the gate.

## Notes

- Help and oil Bayesian CLI remain on `./bgplvm` (see [bgplvm_oil100](./bgplvm_oil100.md)).
- Missing-data views, non-Gaussian likelihoods, and a multi-file `./mrd` CLI are CIP-0018 non-goals for v1.
