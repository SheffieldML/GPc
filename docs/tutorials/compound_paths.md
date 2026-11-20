# Compound kernels and parameter paths

**GPy counterpart:** `basic_kernels` / `models_basic` (composition + constrain)  
**GPc path:** `CCmpndKern` + CIP-0014 `paramPaths` / `setByPath` / `constrainPositive`

Parameter path reference: [param-paths.md](../param-paths.md). Single-output CLI sibling: [regression_1d](./regression_1d.md).

## Build and run

```sh
make demo_compound_paths
./demo_compound_paths
```

Loads `examples/sinc.csv`, builds an RBF + linear compound, prints `paramPaths()`, sets lengthscale/variances by path, restores positive transforms, then short-optimises an FTC GP with `likelihood.variance` observation noise (CIP-0027).

## CLI equivalent (paths only)

```sh
./gp -v 3 learn -# 50 -k rbf -k lin -P rbf.lengthscale=1.0 -P linear.variance=0.1 \
  -P likelihood.variance=0.1 examples/sinc.csv sinc_cmpnd.model
./gp display sinc_cmpnd.model
```

## Notes

- Component labels follow GPy names (`rbf`, `linear`, …); duplicates get `_1`, `_2`, …
- `constrainPositive` / `fix` on paths are the native stand-in for GPy constraint / `Private` patterns.
