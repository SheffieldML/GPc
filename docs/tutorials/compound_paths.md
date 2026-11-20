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

## CLI equivalent

```sh
./gp -v 3 learn -# 50 -k rbf -k lin -P rbf.lengthscale=1.0 -P linear.variance=0.1 \
  -P likelihood.variance=0.1 examples/sinc.csv sinc_cmpnd.model
./gp display sinc_cmpnd.model
```

Matérn / periodic (CIP-000C types on the learn path):

```sh
./gp -v 2 learn -# 30 -k matern32 -P Mat32.lengthscale=1.0 -P Mat32.variance=1.0 \
  -P likelihood.variance=0.1 examples/sinc.csv sinc_m32.model
./gp learn -# 30 -k stdperiodic -P std_periodic.period=2.0 examples/sinc.csv sinc_per.model
```

## Notes

- Component labels follow GPy names (`rbf`, `linear`, `Mat32`, `std_periodic`, …); duplicates get `_1`, `_2`, …
- `constrainPositive` / `fix` on paths are the native stand-in for GPy constraint / `Private` patterns.
- Full `-k` catalogue: `gp learn --help` (kern section).
