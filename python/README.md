# `gpc` — Python façade for GPc (CIP-001D)

Thin NumPy-facing wrapper around the C++ GPc core. Optimisation and kernels stay in C++; this package is an ease-of-use boundary for GPy users ([REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)). Maintainer design notes: [docs/python-facade.md](../docs/python-facade.md) ([CIP-001D](../cip/cip001D.md) / [CIP-002F](../cip/cip002F.md)).

## Install / build

### Users — pip wheel (CIP-002C)

**Distribution name:** `gpc` (import `import gpc`). No CMake / gfortran / system OpenBLAS required for a published wheel.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install gpc                  # from PyPI once published
# or, until the first PyPI release (pin a version that includes the features you need):
#   pip install --no-cache-dir --index-url https://test.pypi.org/simple/ \
#     --extra-index-url https://pypi.org/simple/ "gpc==0.1.0a2"
export OPENBLAS_NUM_THREADS=1
python -c "import gpc; print(gpc.__file__); print(gpc.__version__)"
# optional clean-venv checklist (from a repo checkout):
#   python python/scripts/smoke_installed_gpc.py
```

**Colab / TestPyPI:**

- Keep `--extra-index-url https://pypi.org/simple/` so NumPy resolves from real PyPI (do not point both indexes at TestPyPI).
- Use `--no-cache-dir` after a new TestPyPI upload: pip can keep a stale simple index and report only older alphas (e.g. `from versions: 0.1.0a1` while `0.1.0a2` is already published).
- `0.1.0a1` (2026-11-19) is RBF/ExpQuad only; `0.1.0a2` adds `mlp` / `arccos` and the GPy-form `gpc.kern` surface.

macOS wheels from CI currently require a recent macOS (deployment target follows Homebrew bottle `minOS` on `macos-15`). Linux wheels are manylinux.

### Contributors — editable / source (toolchain required)

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

Windows wheels are Phase D (after CIP-002B).

**Threading:** for wheel installs set `OPENBLAS_NUM_THREADS=1` (and avoid nesting with OpenMP; see [CIP-001F](../cip/cip001F.md) / [docs/build-and-ci.md](../docs/build-and-ci.md)) unless you have measured otherwise. CI wheel tests set this already.

### Publishing (maintainers)

Workflow [`.github/workflows/publish-wheels.yml`](../.github/workflows/publish-wheels.yml):

| Trigger | Behaviour |
| --- | --- |
| `workflow_dispatch` → **artifacts-only** | Build + upload `gpc-wheels` artifact (no PyPI) |
| `workflow_dispatch` → **testpypi** | Build + OIDC publish to TestPyPI (`environment: testpypi`) |
| `workflow_dispatch` → **pypi** or GitHub **Release** | Build + OIDC publish to PyPI (`environment: pypi`); Release also attaches `.whl` assets |

**One-time Trusted Publisher setup** (preferred over API tokens):

1. Create GitHub Environments `testpypi` and `pypi` (optional reviewers / wait timer on `pypi`).
2. On [TestPyPI](https://test.pypi.org/) / [PyPI](https://pypi.org/): Publishing → Trusted Publishers → add pending publisher for `SheffieldML/GPc`, workflow `publish-wheels.yml`, environment `testpypi` or `pypi`.
3. First successful publish creates the `gpc` project; re-run if the pending publisher was added after a failed attempt.

Dry-run without PyPI: Actions → **Publish wheels** → `artifacts-only`, then download `gpc-wheels` and `pip install` a `.whl` into a clean venv + `smoke_installed_gpc.py`.

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

## Interactive help (CIP-002F)

Public constructors and methods carry docstrings for normal Python tooling:

```python
import gpc
help(gpc.GPRegression)
help(gpc.kern.RBF)
m = gpc.GPRegression(X, Y)  # after you have X, Y
help(m.optimize)
```

GPy → `gpc` porting narrative remains in [`docs/gpy_side_by_side.md`](docs/gpy_side_by_side.md) (markdown, not a substitute for `help()`).

## FTC `GPRegression`

GPy-shaped form (preferred for porting):

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
```

Shorthand (string kernel still builds signal + bias + white): `gpc.GPRegression(X, Y)` or `kernel="rbf"` / `"eq"` / `"mlp"` / `"arccos"`. Prefer **noisy** observations: noiseless \(Y\) drives `white.variance` → 0. `objective()` is **−log-likelihood** (SCG minimises it).

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

### MLP / arc-cosine (CIP-001D Phase F)

Williams infinite-MLP (arcsin) and Cho–Saul arc-cosine:

```python
m = gpc.GPRegression(X, Y, kernel="mlp")       # mlp.weight_variance, bias_variance, variance
m = gpc.GPRegression(X, Y, kernel="arccos", arccos_order=0)  # or 1
```

Side-by-side olympic demo: [`examples/mlp_olympic.ipynb`](examples/mlp_olympic.ipynb).

## Sparse `SparseGPRegression`

```python
Z = np.linspace(0.05, 0.95, 8)[:, None]
m = gpc.SparseGPRegression(X, Y, Z)                 # approx="dtcvar" (default)
m = gpc.SparseGPRegression(X, Y, Z, approx="fitc")  # also "dtc"
m = gpc.SparseGPRegression(X, Y, Z, kernel="mlp")   # point inducing + MLP
# CIP-0028 weight-space ridges (MLP / arccos only; Z rows = M):
m = gpc.SparseGPRegression(
    X, Y, Z, kernel="mlp", inducing="weightspace", window_width=0.55
)
m.optimize(iters=50)
```

Inducing inputs `Z` are **fixed** by default (`inducing_fixed=True`). Weight-space compounds are one signal + optional white (no Bias); observation noise stays in β. Tutorial recipes prefer **DTCVAR**; FITC and DTC are available for parity with the C++ API / bench.

## Plotting and notebooks

```python
from gpc.plot import plot_1d  # requires matplotlib
plot_1d(m, X, Y)
```

Examples:

- [`examples/toy_1d_regression.ipynb`](examples/toy_1d_regression.ipynb)
- [`examples/olympic_marathon.ipynb`](examples/olympic_marathon.ipynb) — needs `pods`
- [`examples/mlp_olympic.ipynb`](examples/mlp_olympic.ipynb) — RBF vs MLP (Phase F)
- [`examples/weightspace_ridge_1d.ipynb`](examples/weightspace_ridge_1d.ipynb) — points vs ridges (CIP-0028)

Native CLI/demos remain the primary share format ([CIP-0015](../cip/cip0015.md)). Plotting and `pods` stay **optional extras** (`gpc[plot]`, `gpc[examples]`), not core dependencies.

## Later

CIP-001E live progress / repr, custom kernel factories, coreg / GPLVM façades — follow-ups, not close blockers for the first subset.
