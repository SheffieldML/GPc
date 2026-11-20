# 1D GP regression (sinc)

**GPy counterpart:** `basic_gp` / `GPy.examples.regression.toy_rbf_1d`  
**GPc path:** full GP (`FTC`) + Gaussian noise + default RBF (CIP-0027: observation noise is `likelihood.variance`; Bias/White are opt-in)

## Build

```sh
make gp
```

## CLI tutorial

```sh
./gp -v 3 learn -# 100 examples/sinc.csv sinc.model
./gp display sinc.model
./gp gnuplot -r 400 examples/sinc.csv sinc.model sinc
gnuplot sinc_plot.gp
# or: gnuplot -e "GPLOT_OUT='sinc.png'" sinc_plot.gp
```

`-# 100` caps SCG iterations. Default kernel is RBF only; observation noise defaults to `likelihood.variance=1`. Set parameters by GPy-style path if desired:

```sh
./gp -v 3 learn -# 50 -P rbf.lengthscale=1.0 -P likelihood.variance=0.1 examples/sinc.csv sinc.model
```

Explicit prior Bias/White remain available: `-k rbf -k bias -k white`.

Legacy SVM-light still works: `examples/sinc.svml` (CIP-0022).

## API demo (optional)

```sh
make demo_regression_1d
./demo_regression_1d
```

Loads `examples/sinc.csv`, builds RBF + `likelihood.variance`, sets `rbf.lengthscale` via `setByPath`, runs a short optimisation, and prints the objective. Use the CLI path above for gnuplot figures.

## Notes

- Dense CSV layout: last column is the target `y`; preceding columns are `X` (CIP-0022). `#` comments and an optional header row are allowed.
- Plot colours follow the GPy Tango palette (`CClctrl` gnuplot style).
