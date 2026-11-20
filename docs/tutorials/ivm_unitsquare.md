# IVM classification on the unit square

**GPy counterpart:** `basic_classification` uses Bernoulli + EP. GPc’s documented substitute here is the **IVM** with a probit noise model.  
**Data:** `examples/unitsquaregp.svml` (2D inputs, binary labels)

## Build

```sh
make ivm
```

## CLI tutorial

```sh
./ivm -v 3 learn -a 200 -k rbf examples/unitsquaregp.svml unitsquaregp.model
./ivm display unitsquaregp.model
./ivm gnuplot examples/unitsquaregp.svml unitsquaregp.model unitsquaregp
gnuplot unitsquaregp_plot.gp
# or: gnuplot -e "GPLOT_PNG='unitsquaregp_plot.png'" unitsquaregp_plot.gp
```

`-a 200` sets the active-set size. Default compound is **user kernels + White** (CIP-0027 Phase B′). Example above uses `-k rbf`, so RBF + White.

## Notes

- This is not EP inference inside `CGp`. For teaching, treat IVM+probit as the GPc classification path until a future CIP adds EP/Laplace.
- **IVM softener exception (CIP-0027):** classification softness that is *trained* lives in White-in-\(K\) (`CIvm::updateM` + `getWhite()`). Probit \(\sigma^2\) stays fixed at \(10^{-6}\). Regression / PPA / Python FTC instead put observation noise on `likelihood.variance` and do **not** auto-append White. See [`cip/cip0027/ivm_softener.md`](../../cip/cip0027/ivm_softener.md) and composition results [`cip/cip0027/ivm_kernel_composition.md`](../../cip/cip0027/ivm_kernel_composition.md).
- Kernel Bias is opt-in (`-k bias`); probit bias covers the mean offset.
- Active points (dark red in gnuplot) typically lie near the decision boundary.
