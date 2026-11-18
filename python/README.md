# `gpc` — Python façade for GPc (CIP-001D)

Thin NumPy-facing wrapper around the C++ GPc core. Optimisation and kernels stay in C++; this package is an ease-of-use boundary for GPy users ([REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)).

## Setup (Phase A)

From the repository root (with `make.inc` configured):

```bash
python3 -m venv python/.venv
python/.venv/bin/pip install pybind11 numpy pytest
cd python && make && make test
```

```python
import numpy as np
import gpc

X = np.array([[1.0, 2.0], [3.0, 4.0]])
m = gpc.Matrix.from_numpy(X)
assert m.rows == 2 and m.cols == 2
np.testing.assert_allclose(m.to_numpy(), X)
```

## Phase B (in progress)

```python
import numpy as np
import gpc

X = np.linspace(0, 1, 20)[:, None]
Y = np.sin(2 * np.pi * X)
m = gpc.GPRegression(X, Y)
m.set_by_path("rbf.lengthscale", 0.2)
m.optimize(iters=50)
mu, var = m.predict(X)
```

Default kernel is compound RBF + bias + white. Prefer **noisy** observations in smoke tests: noiseless \(Y\) drives `white.variance` → 0 and near-zero predictive variance at training points (correct, but looks broken). `objective()` is **−log-likelihood** (SCG minimises it).

`optimize(messages=True)` refreshes the paramz-style parameter table **in place**: Jupyter updates one `display_id` cell; terminal / terminal IPython uses ANSI cursor rewrite (not `clear_output`, which stacks in terminal IPython). It does not stream C++ `Iteration:` / gradient-check dumps. Pass `callback=` for outer-iteration hooks.

Custom kernel factories, GPy side-by-side docs, sparse DTCVAR, and Jupyter/`pods` examples are later CIP-001D phases. Plotting and `pods` stay **optional extras** (`gpc[plot]`, `gpc[examples]`), not core dependencies.
