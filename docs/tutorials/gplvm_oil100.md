# GPLVM on oil flow (100 points)

**GPy counterpart:** `GPy.examples.dimensionality_reduction.gplvm_oil_100`  
**Data:** `examples/oilTrain100.svml`

## Build

```sh
make gplvm
```

## CLI tutorial

```sh
./gplvm -v 3 learn -# 100 examples/oilTrain100.svml oil100.model
./gplvm display oil100.model
./gplvm gnuplot oil100.model oil100
gnuplot oil100_plot.gp
# or: gnuplot -e "GPLOT_PNG='oil100_plot.png'" oil100_plot.gp
```

Labels in the SVM-light file colour the latent scatter; they are not used as supervised targets in the default GPLVM likelihood.

## Notes

- Full oil set: `examples/oilTrain.svml` (longer run).
- Swagger/BVH motion-capture tutorials remain deferred (MATLAB mocap helpers).
