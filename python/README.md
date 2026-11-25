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
#     --extra-index-url https://pypi.org/simple/ "gpc==0.1.0a10"
export OPENBLAS_NUM_THREADS=1
python -c "import gpc; print(gpc.__file__); print(gpc.__version__)"
# optional clean-venv checklist (from a repo checkout):
#   python python/scripts/smoke_installed_gpc.py
```

**Colab / TestPyPI:**

- Keep `--extra-index-url https://pypi.org/simple/` so NumPy resolves from real PyPI (do not point both indexes at TestPyPI).
- Use `--no-cache-dir` after a new TestPyPI upload: pip can keep a stale simple index and report only older alphas (e.g. `from versions: 0.1.0a9` while `0.1.0a10` is already published).
- `0.1.0a1` (2026-11-19) is RBF/ExpQuad only; `0.1.0a2` adds `mlp` / `arccos` and the GPy-form `gpc.kern` surface; `0.1.0a3` frees sparse inducing by default and adds optional `learn_window_width` for weight-space; `0.1.0a4` adds CIP-0031 `predict` / `predict_noiseless` and notebook teaching. `0.1.0a5` adds CIP-0032 per-unit weight-space window widths. `0.1.0a6` adds CIP-0036 GPClassification Laplace predict + GPy naming aliases / basic_classification notebook. `0.1.0a7` adds CIP-0038 GPy kern inventory and CIP-0039 GPLVM / BayesianGPLVM / MRD façades. `0.1.0a8` adds CIP-0035 `gpc.models.DeepGP` (DSVI deep GP). `0.1.0a9` adds CIP-0035 DeepGP predict/V7 Jacobian and hidden-width fixes. `0.1.0a10` adds CIP-003A `gpc.likelihoods`, Kern.K/plot, and DeepGP wrap updates.

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

k = gpc.kern.RBF(1)  # CIP-0027: add Bias/White yourself if wanted; noise is likelihood.variance
m = gpc.models.GPRegression(X, Y, kernel=k)
m.set_by_path("rbf.lengthscale", 0.2)
m.optimize(iters=50)
mu, var = m.predict(X)                 # y_* (includes observation noise)
mu_f, var_f = m.predict_noiseless(X)  # f_* (CIP-0031)
```

Shorthand (CIP-0027): string kernels build **signal-only** compounds; observation noise is `likelihood.variance` (alias `Gaussian_noise.variance`). Bias/White are opt-in. Prefer **noisy** observations. `objective()` is **−log-likelihood** (SCG minimises it). `plot_1d(..., noiseless=True)` draws the latent band.

`optimize(messages=True)` shows a paramz-style live view **in place**: with `ipywidgets` (notebook / Colab) a progress bar + status + model HTML update like GPy; without widgets, a single Jupyter `display_id` cell; in a plain terminal, ANSI cursor rewrite. It does not stream C++ `Iteration:` / gradient-check dumps. Pass `callback=` for outer-iteration hooks. Install widgets via `pip install ipywidgets` or `gpc[examples]`.

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

Defaults match GPy (`weight_variance=bias_variance=variance=1`). Large-magnitude inputs need an explicit scale (or standardised `X`); see [`examples/mlp_olympic.ipynb`](examples/mlp_olympic.ipynb).

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

Inducing inputs `Z` (and weight-space ridge centres) are **free** by default (`inducing_fixed=False`, matching C++ `CGp`). Pass `inducing_fixed=True` to freeze a grid. For CIP-0028 ridges, free centres let the finite-\(M\) mean behave like a trainable bank of smoothed erf units. Global window width \(s\) defaults fixed; pass `learn_window_width=True` (with free inducing) to optimise a tied softness, or `learn_window_widths=True` / a length-\(M\) `window_width` array for per-unit \(s_m\) ([CIP-0032](../cip/cip0032.md)). Whitened \(s\) is bounded to \([0.05,1]\) (learned via a sigmoid map) so \(K_{uu}\) stays away from the \(s=\sqrt2\) singular regime. Weight-space compounds are one signal + optional white (no Bias); observation noise stays in \(\beta\). Tutorial recipes prefer **DTCVAR**; FITC and DTC are available for parity with the C++ API / bench.

## Deep GP (`DeepGP`, CIP-0035)

```python
m = gpc.models.DeepGP(X, Y, layers=2, hidden=1, num_inducing=10, chi=0.5)
m.optimize(iters=50, adam_lr=0.05, seed=1)
mu, var = m.predict(X)
m.save("deep.model")
m2 = gpc.models.DeepGP.load("deep.model")
# Probit / ordered finals: likelihood="probit" or "ordered"
# Finite-network bound: drop_residual=True (Nyström residual off)
```

Runnable sketch: [`examples/deepgp_sinc.py`](examples/deepgp_sinc.py). Native CLI: [`docs/tutorials/deepgp.md`](../docs/tutorials/deepgp.md).

## Plotting and notebooks

```python
from gpc.plot import plot_1d  # requires matplotlib
plot_1d(m, X, Y)
```

Examples (GPy-tutorial inspired where noted):

- [`examples/toy_1d_regression.ipynb`](examples/toy_1d_regression.ipynb) — basic GP + `predict` / `predict_noiseless`
- [`examples/sparse_gp_regression_1d.ipynb`](examples/sparse_gp_regression_1d.ipynb) — GPy `sparse_GP_regression_1D`
- [`examples/basic_kernels_compound.ipynb`](examples/basic_kernels_compound.ipynb) — `kern.*` compounds (CIP-0027 noise note)
- [`examples/expquad_regression.ipynb`](examples/expquad_regression.ipynb) — GPy `ExpQuad` path names
- [`examples/olympic_marathon.ipynb`](examples/olympic_marathon.ipynb) — needs `pods`
- [`examples/mlp_olympic.ipynb`](examples/mlp_olympic.ipynb) — RBF vs MLP (Phase F)
- [`examples/weightspace_ridge_1d.ipynb`](examples/weightspace_ridge_1d.ipynb) — points vs ridges (CIP-0028)

Native CLI/demos remain the primary share format ([CIP-0015](../cip/cip0015.md)). Plotting and `pods` stay **optional extras** (`gpc[plot]`, `gpc[examples]`), not core dependencies.

## GPLVM / Bayesian GPLVM / MRD

MAP GPLVM (`CGplvm`), collapsed Bayesian GPLVM (`CBayesGplvm`, CIP-0017), and MRD (`CMrd`, CIP-0018):

```python
Y = ...  # N×D observations
m = gpc.models.GPLVM(Y, input_dim=2)                 # RBF+Bias+White (CLI default)
m.optimize(iters=50)
X = m.latent_mean()                                  # N×q
mu, var = m.predict(X)

m = gpc.models.BayesianGPLVM(Y, input_dim=2, num_inducing=10)  # RBF-ARD
m.optimize(iters=50)
mu_x, s_x = m.latent_mean(), m.latent_variance()

# Multi-view: aligned Y_v (same N), per-view RBF-ARD (GPy MRD default)
m = gpc.models.MRD([Y0, Y1], input_dim=3, num_inducing=10)
m.optimize(iters=50)
fac = m.factorize_space(threshold=0.05)  # shared / private0 / private1
```

Defaults follow GPy’s `kern.RBF` (not `ExpQuad` path names). Native tutorials remain the verified share format ([gplvm_oil100](../docs/tutorials/gplvm_oil100.md), [bgplvm_oil100](../docs/tutorials/bgplvm_oil100.md), [mrd](../docs/tutorials/mrd.md)).

## Later / not in the Python notebook subset yet

Notebook ports of the oil / SVI GPLVM / MRD tutorials are still outstanding. Other GPy-online topics we have **not** mirrored as `gpc` notebooks (coregionalization, sparse EP / IVM, SVGP, SVI BGPLVM, Poisson/hetero/HMC) are listed with owning CIPs in [`docs/subset_checklist.md`](docs/subset_checklist.md). Deep GP is available as `gpc.models.DeepGP` ([CIP-0035](../cip/cip0035.md)). SVI Bayesian GPLVM remains CLI-first today (`./bgplvm --inference svi`); see [docs/tutorials/bgplvm_svi.md](../docs/tutorials/bgplvm_svi.md) and backlog `2026-11-20_cip0030-phase4b-python-facade`.
