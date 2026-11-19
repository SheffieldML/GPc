# 1D GP regression (sinc)

**GPy counterpart:** `basic_gp` / `GPy.examples.regression.toy_rbf_1d`  
**GPc path:** full GP (`FTC`) + Gaussian noise + default RBF (+ bias + white)

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

`-# 100` caps SCG iterations. Default kernel is RBF + bias + white. Set parameters by GPy-style path if desired:

```sh
./gp -v 3 learn -# 50 -P rbf.lengthscale=1.0 examples/sinc.csv sinc.model
```

Legacy SVM-light still works: `examples/sinc.svml` (CIP-0022).

## API demo (optional)

```sh
make demo_regression_1d
./demo_regression_1d
```

Loads `examples/sinc.csv`, builds the same default compound kernel, sets `rbf.lengthscale` via `setByPath`, runs a short optimisation, and prints the objective. Use the CLI path above for gnuplot figures.

## Notes

- Dense CSV layout: last column is the target `y`; preceding columns are `X` (CIP-0022). `#` comments and an optional header row are allowed.
- Plot colours follow the GPy Tango palette (`CClctrl` gnuplot style).
