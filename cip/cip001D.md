---
author: "Neil Lawrence"
created: "2026-11-18"
id: "001D"
last_updated: "2026-11-20"
status: "Closed"
compressed: false
related_requirements: ["000E", "0008", "0005", "0009", "000F"]
related_cips: ["0014", "0015", "0016", "000C", "000D", "000E", "001E", "0028", "002F"]
tags:
- cip
- python
- facade
- gpy
- bindings
- pybind11
- ease-of-use
title: "Light Python façade over GPc for GPy users"
---

# CIP-001D: Light Python façade over GPc for GPy users

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md). It reuses path/unit policy from [CIP-0014](./cip0014.md) / [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md), the kernel catalogue from [CIP-000C](./cip000C.md) / [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md), and the native tutorial subset checklist from [CIP-0015](./cip0015.md) / [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md). Live optimiser progress and structured model repr consume [CIP-001E](./cip001E.md) / [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md). It is **orthogonal** to [CIP-0016](./cip0016.md) (file interchange): this CIP is a live API, not model YAML/JSON.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [x] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Backlog

- [Phase A — scaffold + NumPy/`CMatrix` bridge](../backlog/features/2026-11-18_cip001D-phase-a-scaffold.md) (High, Completed)
- [Phase B — FTC GPRegression + paths](../backlog/features/2026-11-18_cip001D-phase-b-gp.md) (High, Completed)
- [Phase C — sparse DTCVAR façade](../backlog/features/2026-11-18_cip001D-phase-c-sparse.md) (High, Completed)
- [Phase D — subset checklist + polish](../backlog/features/2026-11-18_cip001D-phase-d-polish.md) (Medium, Completed)
- [Phase E — notebooks / pods / thin plot extras](../backlog/features/2026-11-18_cip001D-phase-e-notebooks.md) (Medium, Completed)
- [Phase F — MLP / arc-cosine kernels + olympic notebook](../backlog/features/2026-11-19_cip001D-phase-f-mlp-kernels.md) (High, Completed)
- [GPy-form kern / models / compound +](../backlog/features/2026-11-20_cip001D-gpy-form-kern-models.md) (High, Completed)
- Sibling (CIP-0028): [Python weight-space inducing + ridge notebook](../backlog/features/2026-11-19_cip0028-python-weightspace-notebook.md) (High, Completed)

## Summary

Ship an **installable Python package** that wraps a documented GPc modelling subset behind GPy-like ergonomics: NumPy arrays in/out, path-based parameter get/set in GPy-natural units, `optimize` / `predict` that call the C++ core. Binding uses **pybind11** at the process boundary; SCG, kernels, and sparse algebra stay in C++. Native demos/CLI remain the primary share format for tutorial parity ([CIP-0015](./cip0015.md)); this CIP adds the Python ease-of-use surface GPy users expect.

**Which requirements?** Primary: [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md). Exercises [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md) / [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md) from Python and mirrors the subset checklist of [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md) without replacing it.

## Motivation

GPy users already think in NumPy and short scripts (`GPRegression`, `kern.RBF`, `m.optimize()`, path-ish parameter edits). GPc now has the semantic bridges those users need (GPy path names/units, kernel coverage, sparse DTCVAR), but the only entry points are C++ demos and CLIs. CIP-0015 deliberately deferred Python bindings so native tutorial parity could land first; REQ-000E now owns the outcome. Without a thin façade, “ease of use for GPy users” stops at documentation and file formats ([CIP-0016](./cip0016.md)), and notebook-side labs cannot call GPc without a rewrite.

Ecosystem Interoperability asks us to meet those users in Python. Compose Through Interfaces says the package should be a client of `CKern` / `CNoise` / `CGp`, not a second engine. Flexibility Without a Performance Tax forbids putting the interpreter in the SCG loop. Assert and Verify requires façade vs native (and optional GPy fixture) checks on a tiny seeded problem.

## Detailed Description

### Design decisions (locked for Proposed review)

1. **pybind11 extension module.** One compiled module (working name `_gpc`) plus a thin pure-Python package `gpc` for ergonomics. Rationale: mature NumPy buffer protocol, good C++ class wrapping, familiar to scientific Python. **Rejected for v1:** ctypes/`cffi` only (too much hand marshaling for `CMatrix` and ownership); SWIG (heavier codegen; no current `.i` surface to revive); nanobind (lighter, but smaller ecosystem for contributors copying GPy-era patterns — revisit later if build cost bites); pure Python reimplementation of kernels (violates REQ-000E and Flexibility Without a Performance Tax).
2. **Package name `gpc`.** Import as `import gpc`. Document clearly that this is **not** GPy and does not claim API identity. PyPI namespace / `sheffieldml-gpc` distribution name can differ from the import name if publishing requires it; decide at Phase A without blocking the API sketch.
3. **Ergonomics ≈ GPy recipes, not a paramz clone.** First delivery exposes constructors and methods that feel familiar (`GPRegression`, kernel factories, `optimize`, `predict`, path get/set). It does **not** implement GPy’s full `Parameterized` tree, `__setattr__` magic, or regex `m['.*']`. Paths use the CIP-0014 grammar (`set_by_path` / `get_by_path` / `param_paths`), optionally with a small `m['rbf.lengthscale']` sugar that delegates to that grammar.
4. **NumPy at the boundary only; outer-iteration progress allowed.** `X`, `Y`, and prediction outputs convert to/from `CMatrix` (copy is the default for safety; optional zero-copy later if layout matches and lifetime is proven). **No** Python callbacks inside objective/gradient evaluation, kernel `compute`, or sparse factorisation. **Allowed:** a thin outer-iteration / terminal observer ([CIP-001E](./cip001E.md) `OptimiserEvent`) for `messages=` text and `callback=` (tqdm / notebook widgets). Opt-out is free (`messages=False`, no callback → zero Python traffic during the run). A slow callback stalls the next iteration — same contract as SciPy/`paramz`.
5. **Own the same first subset as the native tutorials.** Map explicitly to CIP-0015’s checklist: FTC Gaussian + compound RBF-style kernel + path get/set; sparse DTCVAR; document “not yet” for coreg / IVM / Bayesian GPLVM until later phases or follow-ups. Coregionalization helpers from CIP-0015 Phase A may be wrapped once they exist natively — not a blocker for Phase B.
6. **Build stays makefile-friendly.** Extension builds against the same BLAS/LAPACK/`make.inc` the rest of GPc uses (via scikit-build-core + CMake, or a documented `python/setup` that links existing objects). Ordinary users install a wheel or `pip install -e python/` after a one-time native build; they do not edit makefiles for day-to-day modelling.
7. **Errors are loud.** Shape mismatches and unknown paths raise Python exceptions that wrap GPc asserts / typed errors; no silent truncation.
8. **Orthogonal to CIP-0016.** Optional later: load/save shared YAML via the façade once CIP-0016 lands. Not required for REQ-000E acceptance.

### API sketch (illustrative; names may tighten in Phase A)

```python
import numpy as np
import gpc

X = np.linspace(0, 1, 20)[:, None]
Y = np.sin(2 * np.pi * X) + 0.05 * np.random.randn(20, 1)

k = gpc.kern.RBF(input_dim=1) + gpc.kern.White(input_dim=1)
m = gpc.models.GPRegression(X, Y, kernel=k)   # FTC + Gaussian noise

m.set_by_path("rbf.lengthscale", 0.2)
m.optimize(max_iters=100, messages=True)   # CIP-001E text observer / default lines
# m.optimize(callback=lambda e: print(e.iter, e.objective))
mu, var = m.predict(X)
print(m)                   # __str__ / __repr__ from ModelDescription (CIP-001E)
print(m.param_paths())
print(m.get_by_path("rbf.variance"))
```

Sparse (Phase C):

```python
m = gpc.models.SparseGPRegression(X, Y, kernel=k, Z=Z, approx="dtcvar")
m.optimize()
```

Python kernel objects are thin handles around `CKern` subclasses (or factory strings that build them in C++). Compound `+` maps to `CCmpndKern`. Parameter paths hit the existing C++ façade — no second naming table in Python. Progress and repr bind CIP-001E; they do not invent a parallel logging or naming path.

### Layout (proposed)

```
python/
  pyproject.toml          # scikit-build-core / package metadata
  gpc/
    __init__.py
    kern.py               # factories / thin wrappers
    models.py             # GPRegression, SparseGPRegression
    _paths.py             # path sugar over C++ API
  src/                    # pybind11 module sources
    module.cpp
    wrap_matrix.cpp
    wrap_kern.cpp
    wrap_gp.cpp
  tests/
    test_ftc_regression.py
    test_paths.py
    test_sparse_dtcvar.py
  docs/
    gpy_side_by_side.md   # GPy → gpc recipes (or compress to docs/ later)
```

Exact directory names can move under `bindings/python/` if that keeps the repo root cleaner; decide in Phase A and document in README.

### Mapping to GPc

| Façade | GPc |
|---|---|
| `numpy.ndarray` | `CMatrix` at call boundary |
| `gpc.kern.*` | `CKern` factories / subclasses already in catalogue |
| `GPRegression` | `CGp` + `CGaussianNoise`, approx FTC |
| `SparseGPRegression` | `CGp` DTCVAR (or documented sparse enum) + inducing `Z` |
| `set_by_path` / `get_by_path` | CIP-0014 path API |
| `optimize` | existing `COptimisable` / SCG (or model’s established train entry) |
| `optimize(..., messages= / callback=)` | CIP-001E outer-iteration `OptimiserEvent` observer (pybind `py::function`) |
| `print(m)` / `__repr__` | CIP-001E `ModelDescription` (+ optional `_repr_html_` later) |
| `predict` | existing posterior mean/variance path |

### Rejected / deferred

| Idea | Status |
|---|---|
| Full GPy/`paramz` object graph | Rejected for v1 (REQ-000E non-goal) |
| Jupyter/Colab as *primary* tutorial share format | Deferred; CIP-0015 remains native-first; optional notebooks are Phase E examples, not the core package |
| Bundling `pods` / matplotlib into core `gpc` | Rejected — keep as optional extras / notebook deps (GPy’s own split) |
| Full GPy `m.plot()` surface | Rejected for v1; optional thin `plot_1d` helper under extras only |
| Reimplementing likelihood in Python | Rejected |
| Python callbacks *inside* objective/gradient / kernel `compute` | Rejected (Flexibility Without a Performance Tax) |
| Outer-iteration live progress / `callback=` | **In scope** via CIP-001E (not a paramz clone) |
| Wrapping IVM / Bayesian GPLVM / MRD in v1 | Deferred to follow-up once FTC+sparse façade is green |
| Putting YAML interchange in this CIP | Owned by CIP-0016 |
| Reviving mex / MATLAB as the Python story | Rejected |

### Layering (locked 2026-11-18)

| Layer | Owns | Must not own |
|---|---|---|
| **`gpc` core** (this CIP Phases A–D) | NumPy ↔ `CMatrix`, models, kernels, paths, `optimize` / `predict` | Dataset download, Jupyter, matplotlib styling |
| **Optional extras** (`gpc[plot]`, later `gpc[examples]`) | Thin helpers e.g. 1-D mean±2σ from `predict` | Full GPy plotting stack; required import for `import gpc` |
| **Notebooks / examples** | `pods.datasets.*`, narrative, figures, marathon/toy recipes | Anything needed for CI of `_gpc` |

`pods` stays an external dependency (same pattern as GPy examples). Notebooks live under `python/examples/` or `docs/notebooks/` and document `pip install pods matplotlib` (or `gpc[examples]`).

### Tenet trade-offs

- **Ecosystem vs Flexibility:** Python is allowed at the boundary for ease of use; hot path stays C++/BLAS. Outer-iteration progress is a rare boundary call, not an inner-loop tax.
- **Compose vs clone:** Prefer fulfilling `CKern`/`CGp` contracts via bindings over mimicking GPy’s inheritance tree; progress/repr compose CIP-001E rather than reimplementing print logic in Python.
- **Assert:** Extension tests must fail closed on bad shapes/paths; do not disable `DIMENSIONMATCH`-class checks to “make NumPy happier.”

## Implementation Plan

1. **Phase A — Scaffold**
   - Add `python/` (or `bindings/python/`) with `pyproject.toml`, pybind11 module stub, and `import gpc` smoke test.
   - Wire build to existing BLAS/LAPACK / object archive; document macOS Accelerate and Linux OpenBLAS paths.
   - Wrap `CMatrix` ↔ NumPy (contiguous `float64`, row-major policy documented).

2. **Phase B — FTC regression + paths**
   - Expose kernel factories needed for the first subset (at least RBF, bias, white, compound add).
   - Expose `GPRegression` (FTC + Gaussian): construct, `optimize`, `predict`, objective readback.
   - Expose CIP-0014 path get/set/list; optional `m[path]` sugar.
   - Side-by-side doc fragment: GPy `basic_gp` / toy 1D → `gpc` script.
   - Pytest: façade vs native C++ on a tiny seeded problem (objective and/or predictions).

3. **Phase B2 — Live progress + repr (depends on CIP-001E Phases A–B)**
   - Bind `OptimiserEvent`; wire `optimize(messages=..., callback=...)`.
   - `__str__` / `__repr__` from `ModelDescription`; optional `_repr_html_` later.
   - Pytest: callback fires on outer iterations; silent when opted out; `str(m)` contains canonical path labels.

4. **Phase C — Sparse DTCVAR**
   - `SparseGPRegression` (or equivalent) with inducing inputs and `dtcvar`.
   - Pytest vs native sparse path; document FITC-not-default (same policy as CIP-0015).

5. **Phase D — Subset map + polish**
   - Checklist table: CIP-0015 tutorial item → `gpc` recipe or “not in Python subset.”
   - Loud error tests (bad shapes, missing paths).
   - README install blurb; optional wheel CI later (not a close blocker if editable install is documented).

6. **Phase E — Notebooks and optional plot extras**
   - Example Jupyter notebooks (toy 1-D; olympic marathon via `pods`) that call core `gpc` only.
   - Optional `gpc[plot]` / `gpc[examples]` extras: thin `plot_1d` over `predict`; `pods` + matplotlib as extra deps, not core.
   - Document noisy-observation smoke recipes (noiseless \(Y\) drives white→0 and looks pathological).
   - Does **not** make Jupyter the primary share format (CIP-0015 remains native-first).

7. **Follow-ups (not this CIP’s close criteria)**
   - Coreg / `build_XY` once CIP-0015 Phase A helpers exist.
   - Bayesian GPLVM / MRD façades.
   - CIP-0016 load/save helpers on the Python object.

## Backward Compatibility

- No change to C++ `.model` streams, CLI flags, or native demos.
- New optional Python package; existing users who ignore `python/` are unaffected.
- Import name `gpc` must not break a future renamed distribution; keep a single public package path from day one.

## Testing Strategy

- **Unit:** NumPy ↔ `CMatrix` round-trip; path get/set natural units; compound kernel construction.
- **Integration:** `pytest` compares façade `optimize`/`predict` to a native `test*` or small C++ harness on fixed seed data within documented tolerances (`ndlutil::MATCHTOL` spirit).
- **Optional:** GPy-generated text fixtures already used elsewhere may gate numerical agreement where the subset overlaps; live GPy import is not required in CI.
- **Negative:** shape mismatch and unknown path raise; no silent clip.

## Related Requirements

- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) — primary WHAT
- [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md) — path/unit policy reused from Python
- [REQ-0005](../requirements/req0005_gpy-aligned-kernel-coverage.md) — kernel catalogue exposed
- [REQ-0009](../requirements/req0009_gpy-tutorial-parity-native.md) — subset checklist mirrored; not replaced
- [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md) — live progress + structured display consumed by Phase B2

## Implementation Status

- [x] Phase A: package scaffold + NumPy/`CMatrix` bridge
- [x] Phase B: FTC `GPRegression` + kernels + paths + side-by-side doc + pytest vs native
- [x] Phase B2: **Deferred** — blocked on [CIP-001E](./cip001E.md) C++ landing; not a close blocker for REQ-000E’s first subset. Bind `messages`/`callback` + `__str__`/`__repr__` when 001E is Implemented.
- [x] Phase C: sparse DTCVAR façade + pytest (`fitc` / `dtc` also exposed; DTCVAR default)
- [x] Phase D: subset checklist, negative tests, README install notes
- [x] Phase E: example notebooks + optional `gpc[plot]`/`gpc[examples]` (pods external)
- [x] Phase F: MLP / arc-cosine string kernels + notebooks
- [x] GPy-form API: `gpc.kern` / `gpc.models` / compound `+` / `kernel=` object (string defaults retained)
- [x] Close: REQ-000E façade acceptance for first subset checked (compound-kernel recipe + side-by-side). Interactive `help()` / docstring richness → [CIP-002F](./cip002F.md). Binary wheels → [CIP-002C](./cip002C.md). Compress markdown into repo `docs/` when convenient (`compressed: false` until then).

## References

- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)
- [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md) / [CIP-001E](./cip001E.md)
- [CIP-0014](./cip0014.md) / [docs/param-paths.md](../docs/param-paths.md)
- [CIP-0015](./cip0015.md) (deferred Python bindings → this CIP)
- [CIP-0016](./cip0016.md) (orthogonal file interchange)
- GPy `models.GPRegression`, `kern.Add`, `paramz` (ergonomics checklist only)
- pybind11 NumPy buffer protocol documentation
