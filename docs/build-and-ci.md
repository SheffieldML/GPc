# Build and CI

Maintainer notes for how GPc builds and how verification is gated. Short commands live in the [README](../README.md); this page holds the durable decisions behind them.

**Design**: [CIP-0001](../cip/cip0001.md) · [CIP-0002](../cip/cip0002.md) · [CIP-0004](../cip/cip0004.md) · [CIP-0006](../cip/cip0006.md)  
**Requirements**: [REQ-0001](../requirements/req0001_contemporary-build.md) · [REQ-0002](../requirements/req0002_runnable-verification.md) · [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md)

## Platforms and make fragments

- Default include is `make.linux` (Ubuntu / Actions). Copy `make.osx` to `make.inc` on Homebrew macOS. `make.inc` is gitignored.
- Blessed macOS path: Homebrew `gcc` / `gfortran` (newest `g++-N` / `gfortran-N`), OpenBLAS, optional `libmatio`.
- Language floor is **C++17** (`-std=c++17`). Dynamic exception specifications were removed so the pin is no longer C++14 ([CIP-0002](../cip/cip0002.md)).
- Windows CI waits on a maintained Windows make path; it is out of scope until that fragment exists.

## Reference verification without MATLAB

Full MATLAB/`mex` is not assumed in CI. Numeric `.mat` fixtures under `matfiles/` load via **libmatio** when built with `USE_MATIO=1` (`_NDLMATIO`).

- Gate: `make USE_MATIO=1 matio-tests` then `make run-matio-tests`.
- Object-level MATLAB I/O (kernel/noise/model structs) remains behind `_NDLMATLAB`.
- Retiring MATLAB bridges entirely is [REQ-0003](../requirements/req0003_matlab-bridge-retirement.md) (deferred).

## GitHub Actions

| Workflow | Role |
| --- | --- |
| **Compile** | Ubuntu + macOS hard gate: build `gp` / `ivm` / `gplvm`, help smoke |
| **Reference tests** | Ubuntu + macOS hard gate: matio suite |
| **Coverage** | Ubuntu only: `COVERAGE=1`, lcov, Codecov upload |

Coverage stays on Ubuntu (gcov/lcov path). `codecov.yml` ignores `test*.cpp` so patch coverage tracks library code. Configure repository secret `CODECOV_TOKEN` for authenticated uploads.

## Ordered categorical missing labels (macOS)

Missing ordered targets are stored as NaN. Casting to `int` before `isnan` is unsafe: on ARM64 `(int)NaN` is `0`, so missing points were scored as category 0. `COrderedNoise` tests `isnan` first. That was a genuine portability bug, not an OpenBLAS mismatch; after the fix, macOS Reference tests are a hard gate ([CIP-0006](../cip/cip0006.md), backlog `2026-11-13_macos-ordered-categorical-noise-mismatch`).
