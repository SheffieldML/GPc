# Sparse 1D GP regression (DTCVAR)

**GPy counterpart:** `sparse_gp_regression` / `SparseGPRegression` (VarDTC-style)  
**GPc path:** `CGp::DTCVAR` — **not** FITC (CLI FITC still errors out)

## Build

```sh
make gp
```

## CLI tutorial

`examples/spgp1d.csv` is a larger 1D set (`.svml` still works). Full GP:

```sh
./gp -v 3 learn -# 100 examples/spgp1d.csv spgp1d.model
```

Sparse variational / DTCVAR with an active set (GPy sparse tutorial analogue):

```sh
./gp -v 3 learn -# 100 -A dtcvar -a 20 examples/spgp1d.csv spgp1d_dtcvar.model
./gp display spgp1d_dtcvar.model
./gp gnuplot -r 400 examples/spgp1d.csv spgp1d_dtcvar.model spgp1d_dtcvar
gnuplot spgp1d_dtcvar_plot.gp
```

`-a` is required for any non-`ftc` approximation. Prefer `-A dtcvar` for tutorials; do not use `-A fitc` until that path is restored.

## API demo (optional)

```sh
make demo_sparse_dtcvar
./demo_sparse_dtcvar
```

Loads `spgp1d.csv`, builds DTCVAR with a fixed inducing subset, sets paths, short optimise, prints objective.

## Notes

- Stable sparse factorisation is the default linear-algebra path ([CIP-000E](../../cip/cip000E.md)); override with `--sparse-linalg legacy` only for diagnostics.
- Inducing inputs can be fixed after init via the model API (`setInducingFixed`); the CLI selects size with `-a`.
