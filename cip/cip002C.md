---
author: "Neil Lawrence"
created: "2026-11-19"
id: "002C"
last_updated: "2026-11-19"
status: "In Progress"  # Phase A workflow landed; soft gate until green on Actions
compressed: false
related_requirements: ["000E", "0007"]
related_cips: ["001D", "0020", "0021", "002B"]
tags:
- cip
- python
- packaging
- wheels
- pypi
- cibuildwheel
- infrastructure
title: "Binary wheels so `pip install` needs no local toolchain"
---

# CIP-002C: Binary wheels so `pip install` needs no local toolchain

> CIPs describe HOW to achieve requirements (WHAT). This CIP completes the packaging story opened by [CIP-0020](./cip0020.md) / [CIP-001D](./cip001D.md): end users get a self-contained wheel so ordinary modelling does not require CMake, gfortran, or a system OpenBLAS install. Native makefile CI stays blessed; this CIP does not reopen [CIP-0021](./cip0021.md) cutover or [CIP-002B](./cip002B.md) Windows CLI smoke. (Numbering note: drafted briefly as 002B locally; on `main`, 002A is SVGP and 002B is Windows MSVC.)

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented (worktree `GPc-cip002C` / `cip002C-binary-wheels`)
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Ship **platform binary wheels** for the `gpc` Python façade via **cibuildwheel** on top of the existing scikit-build-core + thin CMake path, so a GPy user can `pip install` (from PyPI, TestPyPI, or a release asset) and `import gpc` **without** installing CMake, a Fortran compiler, or OpenBLAS by hand. Bundle or repair runtime BLAS/Fortran libs into the wheel. Keep source / editable installs for contributors who already have the native toolchain.

**Which requirements?** [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) acceptance criterion “documented, installable Python package … without … edit[ing] makefiles for ordinary use”; preserve [REQ-0007](../requirements/req0007_competitive-default-build.md) OpenBLAS semantics inside the wheel (same LAPACK-owned linear algebra as CIP-0013 / CIP-0020).

## Motivation

CIP-0020 Phase B made `pip install -e "python/[test]"` work **when** the machine already has CMake, gfortran, and OpenBLAS (and on macOS, Homebrew `CMAKE_PREFIX_PATH` / `g++` / `gfortran`). That meets contributor packaging needs; it does **not** meet REQ-000E for ordinary GPy users.

The remaining gap is binary distribution:

| Path today | Needs toolchain? | Needs system OpenBLAS? |
| --- | --- | --- |
| `python/Makefile` + `libgp` | Yes | Yes |
| `pip install -e python/` (scikit-build) | Yes | Yes |
| **This CIP: platform wheel** | **No** | **No (bundled / repaired)** |

Without wheels, every notebook user re-discovers Fortran and BLAS discovery. Ecosystem Interoperability says meet them with `pip`. Assert and Verify says the wheel must still pass the CIP-001D pytest / FTC smoke. Flexibility Without a Performance Tax is unaffected: the wheel only changes how `_gpc` is delivered, not the SCG inner loop. LAPACK Owns the Linear Algebra still holds — we ship or link OpenBLAS, we do not reimplement GEMM.

## Detailed Description

### Locked decisions

1. **Consume CIP-0020 as-is.** Wheels build through `python/pyproject.toml` → scikit-build-core → root `CMakeLists.txt` with `GPC_BUILD_PYTHON=ON`. Do not invent a second extension build. Makefile remains the blessed native/CI path for C++ gates.

2. **cibuildwheel is the builder.** GitHub Actions matrix produces manylinux (and macOS) wheels; Windows wheels are Phase D after [CIP-002B](./cip002B.md) MSVC+ifx+OpenBLAS smoke is green. No hand-rolled Dockerfiles as the primary path.

3. **Self-contained runtime BLAS (wheel-only exception).** CIP-0020 rejected vendoring BLAS into the *project* CMake for local builds. Wheels are different: end users must not need `apt install libopenblas` / Homebrew OpenBLAS. **Preferred:** build against OpenBLAS in the cibuildwheel image, then **auditwheel** (Linux) / **delocate** or **delocate-fuse** (macOS) / equivalent Windows repair so `libopenblas` and `libgfortran` (or static link where practical) travel with `_gpc*.so` / `.pyd`. Document `OPENBLAS_NUM_THREADS` (and CIP-001F concurrency notes) in the Python README. **Rejected for v1:** runtime dlopen of a system-only OpenBLAS with no repair (breaks “pip only”); shipping a second dense stack (Eigen, MKL-only wheels) as the default.

4. **Fortran stays `ndlfortran.f`.** manylinux / macOS wheel jobs use gfortran; Windows wheel jobs use the same ifx (or documented fallback) story as CIP-002B. Do not revive `ndlfortran.c` / f2c for wheels.

5. **Distribution vs import name.** Import remains `gpc` ([CIP-001D](./cip001D.md)). Before first PyPI upload, check whether the project name `gpc` is available; if not, publish as `sheffieldml-gpc` (or similar) with the same import package. Record the chosen name in `python/pyproject.toml` and README before TestPyPI.

6. **Python versions.** Match current façade: `requires-python >= 3.10`. First matrix: CPython 3.10–3.13 (or whatever cibuildwheel defaults allow on the runners); drop versions that fail for documented ABI reasons rather than blocking the CIP.

7. **Release channel.** Phase C publishes to **TestPyPI** (and/or GitHub Release assets) first; **PyPI** only after a documented smoke (`pip install` into a clean venv, `pytest` or FTC script) and maintainer sign-off. Trusted Publisher (OIDC) preferred over long-lived tokens when available.

8. **Scope limits.** This CIP does **not**: cut over native Compile/Reference/Coverage to CMake ([CIP-0021](./cip0021.md)); replace editable/makefile contributor paths; claim API identity with GPy; bundle `pods` / matplotlib into the core wheel (keep extras). Orthogonal to [CIP-002A](./cip002A.md) (SVGP).

9. **Rejected:** meson as a parallel packaging backend; pure source sdists as the only PyPI artifact without a “needs compiler” warning; Accelerate-only macOS wheels as the sole macOS artifact if OpenBLAS-repaired wheels are feasible (Accelerate may remain a documented *local* CMake option, not the wheel default, to keep parity with CIP-0013 OpenBLAS defaults).

### Effort sketch

| Phase | Scope | Effort |
| --- | --- | --- |
| A | cibuildwheel workflow: Linux + macOS wheels; soft gate; import smoke | ~2–4 days |
| B | auditwheel / delocate (or static) so OpenBLAS + gfortran runtime are in-wheel; pytest on repaired wheel | ~2–4 days |
| C | Docs + TestPyPI / Release assets; PyPI when signed off | ~1–2 days + review |
| D | Windows wheels (after CIP-002B green) | ~2–5 days |

### Alternatives considered

- **Source-only PyPI + “install OpenBLAS first” docs** — fails REQ-000E ordinary-use bar; rejected as the primary story (may remain a documented fallback sdist).
- **Conda-forge only** — valuable later; does not replace pip for GPy notebook users. Orthogonal follow-up, not this CIP’s close criterion.
- **Vendor OpenBLAS into the git tree for all builds** — reopens CIP-0020; rejected. Repair/bundle at *wheel build* time only.
- **Wait for CIP-0021 CMake CI cutover** — orthogonal; wheels can ship while make remains the hard gate.

## Implementation Plan

1. **Phase A — cibuildwheel soft gate**
   - Add `.github/workflows/wheels.yml` (or job) using cibuildwheel against `python/`.
   - Matrix: manylinux x86_64 (+ aarch64 if cheap); macOS (arm64 and/or x86_64 as runners allow).
   - Install/build deps in the image: CMake, gfortran, OpenBLAS headers/libs as required by FindBLAS.
   - Soft gate (`continue-on-error: true`) until green; artifact-upload wheels.
   - Smoke: install wheel into a clean venv; `python -c "import gpc"`.

2. **Phase B — self-contained repair**
   - Configure repair step so `_gpc` does not depend on absolute Homebrew / `/usr/lib` OpenBLAS paths (today’s Apple `INSTALL_RPATH` in `python/CMakeLists.txt` is contributor-oriented; wheels need `@loader_path` / `$ORIGIN`-relative layout after delocate/auditwheel).
   - Run `python/tests` (or a documented FTC subset) against the repaired wheel in CI.
   - Document thread-env vars and any known OpenBLAS symbol constraints.

3. **Phase C — publish path**
   - Confirm PyPI project name; update `project.name` if needed.
   - Document: `pip install gpc` (or `sheffieldml-gpc`) for users; keep `pip install -e "python/[test]"` for contributors in `python/README.md` and root README.
   - TestPyPI (or GitHub Release) dry-run; then PyPI with Trusted Publisher if feasible.
   - Versioning: keep façade version in `pyproject.toml`; align tags with release process (no need to version the C++ CLI the same way).

4. **Phase D — Windows wheels**
   - Blocked on CIP-002B compile smoke being green (soft→hard optional).
   - cibuildwheel Windows + MSVC/ifx/OpenBLAS; repair DLLs onto `PATH` / wheel package data.
   - Same import + pytest smoke as Unix.

## Backward Compatibility

- Additive for end users: new install path; no change to `import gpc` API from CIP-001D.
- Contributors keep makefile and editable scikit-build paths.
- No change to `.model` streams, native CLIs, or matio gates.
- If the PyPI distribution name differs from `gpc`, document `pip install sheffieldml-gpc` → `import gpc` once; do not rename the import package.

## Testing Strategy

- **Phase A**: wheel build succeeds; `import gpc` in clean venv on each OS/arch in the matrix.
- **Phase B**: `pytest -q python/tests` (or FTC smoke) on the *repaired* wheel; ldd/otool shows no unresolved absolute OpenBLAS from Homebrew system prefixes.
- **Phase C**: clean-machine checklist (or Actions job): `pip install` from TestPyPI → import → tiny `GPRegression` optimize/predict.
- **Phase D**: same on Windows after CIP-002B.
- Native make Compile / Reference / Coverage remain the merge hard gates unless CIP-0021 says otherwise; wheel workflow starts soft.

## Related Requirements

- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) — installable façade without makefile edits for ordinary use.
- [REQ-0007](../requirements/req0007_competitive-default-build.md) — OpenBLAS / competitive defaults inside the shipped binary.

Specifically, this CIP implements:
- Binary wheels via cibuildwheel on the CIP-0020 scikit-build path
- In-wheel BLAS/Fortran runtime so pip users need no system maths stack
- Documented TestPyPI / PyPI (or Release asset) install story

## Implementation Status

- [x] Phase A: cibuildwheel Linux + macOS soft gate + import smoke + artifacts (`.github/workflows/wheels.yml`; `continue-on-error: true` until green). macOS skips delocate until Phase B (Homebrew bottles are minOS 26 on `macos-latest`).
- [x] Phase B: auditwheel/delocate self-contained OpenBLAS + gfortran; pytest + `check_native_deps.py` on repaired wheel; macOS deployment target aligned to brew bottle minOS (`macos-15`)
- [ ] Phase C: docs + TestPyPI/Release; PyPI after sign-off; distribution name locked
- [ ] Phase D: Windows wheels after CIP-002B

## References

- [CIP-001D](./cip001D.md) — façade API; import `gpc`; optional distinct PyPI name
- [CIP-0020](./cip0020.md) — thin CMake + scikit-build Phase B (closed)
- [CIP-0021](./cip0021.md) — CMake CI cutover (orthogonal)
- [CIP-002B](./cip002B.md) — Windows MSVC+ifx smoke (Windows wheels depend on this)
- Packaging surface: `python/pyproject.toml`, `python/CMakeLists.txt`, `python/README.md`
- Tools: [cibuildwheel](https://cibuildwheel.pypa.io/), auditwheel, delocate; scikit-build-core
- Delivery worktree: `GPc-cip002C` / branch `cip002C-binary-wheels`
