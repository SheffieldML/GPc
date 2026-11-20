# GPy → `gpc` side-by-side (CIP-001D)

Short porting map for the first subset: FTC Gaussian regression with a compound kernel. Native CLI/demos remain the primary tutorial share format ([CIP-0015](../../cip/cip0015.md) / [`docs/tutorials/regression_1d.md`](../../docs/tutorials/regression_1d.md)); this page is for GPy users calling the Python façade.

## Toy 1-D (`basic_gp` / `toy_rbf_1d`)

**GPy**

```python
import numpy as np
import GPy

X = np.linspace(0, 1, 20)[:, None]
Y = np.sin(2 * np.pi * X) + 0.05 * np.random.randn(20, 1)

k = GPy.kern.RBF(1) + GPy.kern.Bias(1) + GPy.kern.White(1)
m = GPy.models.GPRegression(X, Y, k)
m["rbf.lengthscale"] = 0.2
m.optimize()
mu, var = m.predict(X)
```

**`gpc` (kern / models form)**

```python
import numpy as np
import gpc

X = np.linspace(0, 1, 20)[:, None]
Y = np.sin(2 * np.pi * X) + 0.05 * np.random.randn(20, 1)

k = gpc.kern.RBF(1) + gpc.kern.Bias(1) + gpc.kern.White(1)
m = gpc.models.GPRegression(X, Y, kernel=k)
m.set_by_path("rbf.lengthscale", 0.2)
m.optimize(iters=50)
mu, var = m.predict(X)
print(m.param_paths())
print(m.objective())  # −log-likelihood (SCG minimises this)
```

**Shorthand (default compound):** `gpc.GPRegression(X, Y)` still builds signal + bias + white for `kernel='rbf'` / `'eq'` / `'mlp'` / `'arccos'` string names.

Install / build: see [`../README.md`](../README.md).

## Notes that bite when porting

| Topic | GPy | `gpc` |
|---|---|---|
| Kernel construction | `GPy.kern.*` + `+` | `gpc.kern.*` + `+` → exact compound; string `kernel=` keeps auto Bias/White |
| Parameter access | `m["rbf.lengthscale"]` / paramz | `set_by_path` / `get_by_path` / `param_paths` (CIP-0014 units) |
| Objective | log-likelihood (maximise) | `objective()` is **−log-likelihood** (minimise) |
| Noise | Often `Gaussian` likelihood variance | Prefer observation noise via `White` (or string-default white); noiseless \(Y\) drives white→0 |
| Sparse | `SparseGPRegression` / VarDTC | `gpc.models.SparseGPRegression(..., Z=Z, kernel=k)` → default `dtcvar`; also `approx="fitc"` / `"dtc"` |

## CLI twin

Same modelling surface without Python:

```sh
./gp -v 3 learn -# 100 -P rbf.lengthscale=1.0 examples/sinc.svml sinc.model
```

See [`docs/tutorials/regression_1d.md`](../../docs/tutorials/regression_1d.md).
