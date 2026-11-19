# `gpc` — Python façade for GPc (CIP-001D)

Thin NumPy-facing wrapper around the C++ GPc core. Optimisation and kernels stay in C++; this package is an ease-of-use boundary for GPy users ([REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)).

## Install / build

### Makefile path (day-to-day, CIP-001D)

From the repository root (with `make.inc` configured — copy `make.osx` or `make.linux`):

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
pip install pybind11 numpy pytest
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

### Binary wheels (CIP-002C)

GitHub Actions workflow [`.github/workflows/wheels.yml`](../.github/workflows/wheels.yml) builds manylinux and macOS wheels with **cibuildwheel** (soft gate until soak). Config lives in this `pyproject.toml` under `[tool.cibuildwheel]`.

- **Linux:** `auditwheel` vendors OpenBLAS + libgfortran into the wheel.
- **macOS:** `delocate` vendors the same; `MACOSX_DEPLOYMENT_TARGET` is aligned to Homebrew bottle `minOS` on the runner (currently `macos-15` → typically 15.x). Wheels are self-contained (no Homebrew at import). Lowering the macOS floor further needs rebuilding OpenBLAS/gfortran from source, not stock brew bottles.
- **CI checks:** `python/scripts/check_native_deps.py` plus `pytest` on the repaired wheel.

Ordinary users will eventually `pip install` a published wheel (Phase C); until then, use the editable path above. Windows wheels are Phase D (after CIP-002B).

**Threading:** for wheel installs set `OPENBLAS_NUM_THREADS=1` (and avoid nesting with OpenMP; see [CIP-001F](../cip/cip001F.md) / [docs/build-and-ci.md](../docs/build-and-ci.md)) unless you have measured otherwise. CI wheel tests set this already.

Optional extras (not required for `import gpc` or core tests):

```bash
pip install matplotlib          # gpc[plot]
pip install pods jupyter        # notebooks / examples
# or: pip install -e "python/[plot,examples,test]"
```

```python
import numpy as np
import gpc

X = np.array([[1.0, 2.0], [3.0, 4.0]])
m = gpc.Matrix.from_numpy(X)
assert m.rows == 2 and m.cols == 2
np.testing.assert_allclose(m.to_numpy(), X)
```

## FTC `GPRegression`

```python
import numpy as np
import gpc

X = np.linspace(0, 1, 20)[:, None]
Y = np.sin(2 * np.pi * X) + 0.05 * np.random.randn(20, 1)
m = gpc.GPRegression(X, Y)
m.set_by_path("rbf.lengthscale", 0.2)
m.optimize(iters=50)
mu, var = m.predict(X)
```

Default kernel is compound RBF + bias + white. Prefer **noisy** observations: noiseless \(Y\) drives `white.variance` → 0. `objective()` is **−log-likelihood** (SCG minimises it).

- GPy porting fragment: [`docs/gpy_side_by_side.md`](docs/gpy_side_by_side.md)
- Tutorial map: [`docs/subset_checklist.md`](docs/subset_checklist.md)
- Native objective compare: `make test` builds `ref_ftc_objective`

### GPy ExpQuad → `gpc` (CIP-0024)

GPy’s lengthscale-first squared exponential is `ExpQuad`. Pass `kernel="eq"` (aliases `ExpQuad` / `expquad`) so compound paths use `ExpQuad.*` instead of `rbf.*`:

```python
m = gpc.GPRegression(X, Y, kernel="eq")
m.set_by_path("ExpQuad.lengthscale", 0.2)
m.optimize(iters=50)
```

Runnable sketch: [`examples/expquad_regression.py`](examples/expquad_regression.py). Same \(K_of_r\) as RBF; storage remains `inverseWidth` under the hood ([docs/param-paths.md](../docs/param-paths.md)).

## Sparse `SparseGPRegression`

```python
Z = np.linspace(0.05, 0.95, 8)[:, None]
m = gpc.SparseGPRegression(X, Y, Z)                 # approx="dtcvar" (default)
m = gpc.SparseGPRegression(X, Y, Z, approx="fitc")  # also "dtc"
m.optimize(iters=50)
```

Inducing inputs `Z` are **fixed** by default. Tutorial recipes prefer **DTCVAR**; FITC and DTC are available for parity with the C++ API / bench.

## Plotting and notebooks

```python
from gpc.plot import plot_1d  # requires matplotlib
plot_1d(m, X, Y)
```

Examples (Phase E):

- [`examples/toy_1d_regression.ipynb`](examples/toy_1d_regression.ipynb)
- [`examples/olympic_marathon.ipynb`](examples/olympic_marathon.ipynb) — needs `pods`

Native CLI/demos remain the primary share format ([CIP-0015](../cip/cip0015.md)). Plotting and `pods` stay **optional extras** (`gpc[plot]`, `gpc[examples]`), not core dependencies.

## Later

CIP-001E live progress / repr, custom kernel factories, coreg / GPLVM façades — follow-ups, not close blockers for the first subset.
