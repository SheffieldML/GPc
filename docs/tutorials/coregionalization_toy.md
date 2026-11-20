# Coregionalization toy (two sinusoids)

**GPy counterpart:** `GPy.examples.regression.coregionalization_toy` / `coregionalized_regression_tutorial`  
**GPc path:** `CCoregionalizedRegression` (ICM + `indexwhite`, FTC) via [`CMultioutput`](../coregionalize.md)

Multi-output layout and helpers: [coregionalize.md](../coregionalize.md). There is no multi-output `./gp` CLI yet — use the API demo.

## Build and run

```sh
make demo_coregionalization_toy
./demo_coregionalization_toy
```

The demo synthesises two noisy sinusoids (GPy sizes: \(N_0=50\) on \([0,8]\), \(N_1=30\) on \([0,5]\), second output offset by \(+2\)), builds an ICM RBF coregionalized GP with per-output `indexwhite`, runs a short optimisation, prints slice ranges, and predicts on a shared spatial grid with `stampOutputIndex` for each output.

## What to expect

- Stacked design from `buildXY` (index column last).
- Per-output noise in `indexwhite`; spherical `likelihood.variance` fixed tiny (CIP-0027).
- Finite objective and per-output predictive means after ~25 SCG steps.

Unit coverage of helpers/factory: `make testCoregionalize && ./testCoregionalize`.

## Notes

- LCM: pass a list of spatial kernels into the LCM `CCoregionalizedRegression` constructor (see [coregionalize.md](../coregionalize.md)).
- Plotting: dump `posteriorMeanVar` tables and use gnuplot; CLI `gp gnuplot` stays single-output.
