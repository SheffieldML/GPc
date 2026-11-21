# Python façade (`gpc`)

Maintainer decisions for the NumPy-facing wrapper. Install and API recipes live in [`python/README.md`](../python/README.md); this page records why the surface looks the way it does.

**Design**: [CIP-001D](../cip/cip001D.md) · [CIP-002F](../cip/cip002F.md) · [CIP-0020](../cip/cip0020.md) · [CIP-002C](../cip/cip002C.md) · [CIP-0033](../cip/cip0033.md) · [CIP-0034](../cip/cip0034.md)  
**Requirements**: [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) · [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md) · [REQ-0019](../requirements/req0019_nonconjugate-inference-ep-laplace.md)

## Role

`gpc` is an ease-of-use boundary for GPy users: NumPy in/out, familiar constructors (`GPRegression`, `gpc.kern.*`), path get/set, `optimize` / `predict`. SCG, kernels, and sparse algebra stay in C++. It is **not** a paramz clone and does not claim GPy API identity. Native CLI/demos remain the primary share format for tutorial parity ([CIP-0015](../cip/cip0015.md)); file interchange is [CIP-0016](../cip/cip0016.md) (orthogonal).

## Locked decisions ([CIP-001D](../cip/cip001D.md))

1. **pybind11** extension `_gpc` plus thin pure-Python `gpc` package.
2. **Import name `gpc`**; distribution name on TestPyPI/PyPI is also `gpc` ([CIP-002C](../cip/cip002C.md)).
3. **Ergonomics ≈ GPy recipes**, not Parameterized/`__setattr__` magic. Paths use the CIP-0014 grammar; optional `m['rbf.lengthscale']` sugar delegates to it.
4. **NumPy at the boundary only.** No Python callbacks inside objective, gradient, kernel compute, or sparse factorisation. Outer-iteration observers ([CIP-001E](../cip/cip001E.md)) are allowed; opt-out is free.
5. **Same modelling subset as native tutorials** first (FTC, sparse DTCVAR, MLP/arccos, GPy-form `kern` / compound `+`). Loud errors on shape/path mistakes.
6. **Build:** day-to-day editable via `python/Makefile` against root `libgp`; wheels via scikit-build-core + thin CMake ([CIP-0020](../cip/cip0020.md) / [CIP-002C](../cip/cip002C.md)). Make remains the blessed native/CI path.

Deferred from 001D close: richer CIP-001E optimiser/model display landed with [CIP-001E](../cip/cip001E.md). Broader GPy-tutorial notebooks still out of subset (coreg, IVM sparse classification, GPLVM/MRD, SVGP, SVI BGPLVM) — see [`python/docs/subset_checklist.md`](../python/docs/subset_checklist.md).

## Classification ([CIP-0033](../cip/cip0033.md) / [CIP-0034](../cip/cip0034.md) / [CIP-0036](../cip/cip0036.md))

- **`gpc.GPClassification(X, Y, kernel=..., inference="ep"|"laplace")`** — dense FTC Bernoulli–probit; EP is the default GPy path; Laplace optional with full latent predict (CIP-0036).
- Predictive: `predict` / `predict_noiseless` → latent \(f_*\); `predict_proba` / `predictive_probabilities` → \(P(y=1)\). Sparse EP / EPDTC deferred.
- Teaching: [`python/examples/basic_classification.ipynb`](../python/examples/basic_classification.ipynb); capability matrix in [CIP-0036](../cip/cip0036.md) / [gpy_side_by_side](../python/docs/gpy_side_by_side.md).
- C++: `CGp::setEpInference` / `setLaplaceInference` / `laplaceLatentMeanVar`; tutorial [cgp_ep_classification](./tutorials/cgp_ep_classification.md). Large-\(N\) active-set remains IVM ([ivm_unitsquare](./tutorials/ivm_unitsquare.md)).
- Likelihood site caches are not façade-visible; ADF state lives in inference workspaces ([CIP-0034](../cip/cip0034.md)).

## Help and docstrings ([CIP-002F](../cip/cip002F.md))

Interactive discoverability is a **separate** requirement from the porting narrative:

- Source of truth for REPL/IDE help is `__doc__` on the public surface (`help(gpc.GPRegression)`, IPython `?`).
- Markdown side-by-side guides stay the GPy porting story ([`python/docs/`](../python/docs/)).
- Cover the documented REQ-000E subset; expand when new façade surfaces land.
- Sphinx autodoc may consume the same strings later; it is not a close blocker.
- Smoke: `python/tests/test_help_docstrings.py`.

## Where to read next

| Need | Location |
|---|---|
| Install (wheel / editable / Colab TestPyPI) | [`python/README.md`](../python/README.md) |
| GPy porting map | [`python/docs/gpy_side_by_side.md`](../python/docs/gpy_side_by_side.md) |
| Subset checklist | [`python/docs/subset_checklist.md`](../python/docs/subset_checklist.md) |
| Classification tutorial | [tutorials/cgp_ep_classification.md](./tutorials/cgp_ep_classification.md) |
| Architecture (likelihood / sites) | [target-architecture.md](./target-architecture.md) |
| CMake / wheels / OpenBLAS threading | [build-and-ci.md](./build-and-ci.md) |
