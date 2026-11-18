# `gpc` — Python façade for GPc (CIP-001D)

Thin NumPy-facing wrapper around the C++ GPc core. Optimisation and kernels stay in C++; this package is an ease-of-use boundary for GPy users ([REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)).

## Setup

### Makefile path (day-to-day, CIP-001D)

From the repository root (with `make.inc` configured):

```bash
python3 -m venv python/.venv
python/.venv/bin/pip install pybind11 numpy pytest
cd python && make && make test
```

### scikit-build / editable install (CIP-0020 Phase B)

Builds `_gpc` via the repo-root thin CMake (OpenBLAS + Fortran). Make remains the blessed native/CI path; this is the packaging-oriented path.

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
# Homebrew macOS: point CMake at OpenBLAS (and prefer g++/gfortran like make.osx).
export CMAKE_PREFIX_PATH="$(brew --prefix openblas)"
export CXX="$(/bin/ls "$(brew --prefix gcc)/bin"/g++-[0-9]* | sort -V | tail -1)"
export FC="$(/bin/ls "$(brew --prefix gcc)/bin"/gfortran-[0-9]* | sort -V | tail -1)"
pip install -e "python/[test]"
pytest -q python/tests
```

`python/Makefile` stays supported for editable development without scikit-build.

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

Default kernel is compound RBF + bias + white. Prefer **noisy** observations in smoke tests: noiseless \(Y\) drives `white.variance` → 0 and near-zero predictive variance at training points (correct, but looks broken). `objective()` is **−log-likelihood** (SCG minimises it). SCG may print a max-iteration warning when the budget is hit.

### GPy ExpQuad → `gpc` (CIP-0024)

GPy’s lengthscale-first squared exponential is `ExpQuad`. Pass `kernel="eq"` (aliases `ExpQuad` / `expquad`) so compound paths use `ExpQuad.*` instead of `rbf.*`:

```python
# GPy sketch:
# k = GPy.kern.ExpQuad(1) + GPy.kern.Bias(1) + GPy.kern.White(1)
# m = GPy.models.GPRegression(X, Y, k)
# m.ExpQuad.lengthscale = 0.2

m = gpc.GPRegression(X, Y, kernel="eq")
m.set_by_path("ExpQuad.lengthscale", 0.2)
m.optimize(iters=50)
```

Runnable sketch: [`examples/expquad_regression.py`](examples/expquad_regression.py). Same \(K_of_r\) as RBF; storage remains `inverseWidth` under the hood ([docs/param-paths.md](../docs/param-paths.md)).

Custom kernel factories, sparse DTCVAR, and Jupyter/`pods` examples are later CIP-001D phases. Plotting and `pods` stay **optional extras** (`gpc[plot]`, `gpc[examples]`), not core dependencies.
