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
mu_f, var_f = m.predict_noiseless(X)
```

**`gpc` (kern / models form)**

```python
import numpy as np
import gpc

X = np.linspace(0, 1, 20)[:, None]
Y = np.sin(2 * np.pi * X) + 0.05 * np.random.randn(20, 1)

k = gpc.kern.RBF(1)  # CIP-0027: Bias/White opt-in; noise is likelihood.variance
m = gpc.models.GPRegression(X, Y, kernel=k)
m.set_by_path("rbf.lengthscale", 0.2)
m.optimize(iters=50)
mu, var = m.predict(X)  # y_* (includes observation noise)
mu_f, var_f = m.predict_noiseless(X)  # f_* (CIP-0031)
print(m.param_paths())
print(m.objective())  # −log-likelihood (SCG minimises this)
```

**Shorthand:** string `kernel='rbf'` / `'eq'` / `'mlp'` / `'arccos'` builds a **signal-only** compound; observation noise is `likelihood.variance` (CIP-0027).

Install / build: see [`../README.md`](../README.md).

## Notes that bite when porting

| Topic | GPy | `gpc` |
|---|---|---|
| Kernel construction | `GPy.kern.*` + `+` | `gpc.kern.*` + `+` → exact compound; string `kernel=` is signal-only (CIP-0027) |
| Kernel matrix / plot | `k.K(X)`, `k.plot()` | `k.K(X[, X2])`; thin `k.plot` / `gpc.plot.plot_kern` (`gpc[plot]`) — not full GPy widgets |
| Parameter access | `m["rbf.lengthscale"]` / paramz | `set_by_path` / `get_by_path` / `param_paths` (CIP-0014 units) |
| Objective | log-likelihood (maximise) | `objective()` is **−log-likelihood** (minimise) |
| Noise | `likelihood.variance` / `Gaussian_noise.variance` | Same paths (CIP-0027); sparse algebra may keep internal \(\beta\) |
| Predict | `predict` / `predict_noiseless` | Same names (CIP-0031): `y_*` vs `f_*`; see [`examples/toy_1d_regression.ipynb`](../examples/toy_1d_regression.ipynb) |
| Sparse | `SparseGPRegression` / VarDTC | `gpc.models.SparseGPRegression(..., Z=Z, kernel=k)` → default `dtcvar`; also `approx="fitc"` / `"dtc"` |

## Binary classification (`basic_classification`)

**GPy**

```python
import numpy as np
import GPy

X = np.linspace(-2, 2, 40)[:, None]
Y = (np.sin(X[:, 0]) > 0).astype(float)[:, None]

m = GPy.models.GPClassification(X, Y)  # Bernoulli + EP
m.optimize()
mu, var = m.predict_noiseless(X)       # latent f_*
p = m.predictive_probabilities(X)      # P(y=1)
```

**`gpc`**

```python
import numpy as np
import gpc

X = np.linspace(-2, 2, 40)[:, None]
Y = (np.sin(X[:, 0]) > 0).astype(float)[:, None]

m = gpc.GPClassification(X, Y, inference="ep")  # or "laplace"
m.optimize(iters=60)
mu, var = m.predict_noiseless(X)   # also m.predict(X) — latent f_*
p = m.predictive_probabilities(X)  # alias of predict_proba
```

| Topic | GPy | `gpc` |
|---|---|---|
| Default inference | EP | `inference="ep"` |
| Laplace | optional | `inference="laplace"` (predict wired, CIP-0036) |
| Latent predictive | `predict` / `predict_noiseless` | Same (`predict` aliases `predict_noiseless`) |
| Class probabilities | `predictive_probabilities` | `predict_proba` **and** `predictive_probabilities` |
| White softener | not on default classifier | No auto White (CIP-0027); sites/Hessian hold softness |
| Sparse EP / EPDTC | available in GPy | **Deferred** (FTC-only; CIP-0033/0036) |
| Large \(N\) | sparse classifiers | Native IVM tutorial; not wrapped in `gpc` yet |

Notebook: [`examples/basic_classification.ipynb`](../examples/basic_classification.ipynb). Native twin: [`docs/tutorials/cgp_ep_classification.md`](../../docs/tutorials/cgp_ep_classification.md). Full capability matrix: [CIP-0036](../../cip/cip0036.md).

## CLI twin

Same modelling surface without Python:

```sh
./gp -v 3 learn -# 100 -P rbf.lengthscale=1.0 examples/sinc.svml sinc.model
```

See [`docs/tutorials/regression_1d.md`](../../docs/tutorials/regression_1d.md).
