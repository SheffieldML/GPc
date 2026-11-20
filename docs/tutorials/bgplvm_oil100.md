# Bayesian GPLVM on oil flow (100 points)

**GPy counterpart:** `GPy.models.BayesianGPLVM` / oil-style latent visualisation  
**Data:** `examples/oilTrain100.svml`  
**Design:** [CIP-0017](../../cip/cip0017.md) · [REQ-000B](../../requirements/req000B_bayesian-gplvm-psi.md)

MAP sibling: [gplvm_oil100](./gplvm_oil100.md) (`./gplvm`).

## Build

```sh
make bgplvm
```

## CLI tutorial

```sh
./bgplvm -v 2 learn -x 2 -M 10 -# 100 -o oil100_bayes \
    examples/oilTrain100.svml oil100_bayes.model
./bgplvm display oil100_bayes.model
./bgplvm gnuplot oil100_bayes.model oil100_bayes
gnuplot oil100_bayes_plot.gp
# or: gnuplot -e "GPLOT_OUT='oil100_bayes.png'" oil100_bayes_plot.gp
```

Default kernel is **RBF-ARD** (`rbfard`) so unused latent dims can shut off (GPy `BayesianGPLVM` default). Override with `-k rbf`, or `-k rbf -i true` to match the MAP `./gplvm` compound defaults.

`-o` writes `oil100_bayes_mu.dat` and `oil100_bayes_S.dat` (variational mean and diagonal variance of \(q(\mathbf{X})\)). `gnuplot` plots \(\boldsymbol{\mu}\) (coloured by SVM-light labels when present) over a predictive-variance background. For \(q>2\), choose the plane with `-d i j` (zero-based; default `0 1`).

Optimiser budget `-#` is algorithm iterations for SCG (default). See [optimisers.md](../optimisers.md) for L-BFGS vs SCG semantics.

## Notes

- Full oil set: `examples/oilTrain.svml` (longer run).
- Uncollapsed SVI path (no psi / large \(N\)): [bgplvm_svi](./bgplvm_svi.md) (`--inference svi`).
- Multi-view extension: [mrd](./mrd.md) (`CMrd` / `demo_mrd`).
