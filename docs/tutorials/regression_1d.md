# 1D GP regression (sinc)

**GPy counterpart:** `basic_gp` / `GPy.examples.regression.toy_rbf_1d`  
**GPc path:** full GP (`FTC`) + Gaussian noise + default RBF (+ bias + white)

## Build

```sh
make gp
```

## CLI tutorial

```sh
./gp -v 3 learn -# 100 examples/sinc.svml sinc.model
./gp display sinc.model
./gp gnuplot -r 400 examples/sinc.svml sinc.model sinc
gnuplot sinc_plot.gp
# or: gnuplot -e "GPLOT_OUT='sinc.png'" sinc_plot.gp
```

`-# 100` caps SCG iterations. Default kernel is RBF + bias + white. Set parameters by GPy-style path if desired:

```sh
./gp -v 3 learn -# 50 -P rbf.lengthscale=1.0 examples/sinc.svml sinc.model
```

## API demo (optional)

```sh
make demo_regression_1d
./demo_regression_1d
```

Loads `examples/sinc.svml`, builds the same default compound kernel, sets `rbf.lengthscale` via `setByPath`, runs a short optimisation, and prints the objective. Use the CLI path above for gnuplot figures.

## Notes

- Labels in the SVM-light file are for visualisation; regression uses the continuous targets.
- Plot colours follow the GPy Tango palette (`CClctrl` gnuplot style).
