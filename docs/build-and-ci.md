# Build and CI

Maintainer notes for how GPc builds and how verification is gated. Short commands live in the [README](../README.md); this page holds the durable decisions behind them.

**Design**: [CIP-0001](../cip/cip0001.md) · [CIP-0002](../cip/cip0002.md) · [CIP-0004](../cip/cip0004.md) · [CIP-0006](../cip/cip0006.md) · [CIP-0010](../cip/cip0010.md) · [CIP-0011](../cip/cip0011.md) · [CIP-0013](../cip/cip0013.md) · [CIP-0020](../cip/cip0020.md) · [CIP-0025](../cip/cip0025.md)  
**Requirements**: [REQ-0001](../requirements/req0001_contemporary-build.md) · [REQ-0002](../requirements/req0002_runnable-verification.md) · [REQ-0004](../requirements/req0004_modern-cpp-toolchain.md) · [REQ-0007](../requirements/req0007_competitive-default-build.md)

## Platforms and make fragments

- **Blessed path is still make.** Default include is `make.linux` (Ubuntu / Actions). Copy `make.osx` to `make.inc` on Homebrew macOS. `make.inc` is gitignored. Compile / Reference / Coverage CI use this path.
- **Optional thin CMake** ([CIP-0020](../cip/cip0020.md)): `cmake -B build -DCMAKE_PREFIX_PATH="$(brew --prefix openblas)" && cmake --build build` builds `libgp` and CLI smoke targets (`gp`, and optionally `ivm` / `gplvm` / `bgplvm`). On macOS, pass Homebrew `g++-N` / `gfortran-N` (or rely on `GPC_PREFER_HOMEBREW_GCC`) so C++ and Fortran ABIs match `make.osx`. `CMAKE_BUILD_TYPE=Release` maps to `make RELEASE=1` checks-off semantics; `-DGPC_USE_MATIO=ON` maps to `_NDLMATIO`. Sources are listed in [`cmake/GpcSources.cmake`](../cmake/GpcSources.cmake). CMake is **not** the CI gate until Phase C parity.
- **Python packaging (Phase B):** from `python/`, `pip install -e ".[test]"` uses scikit-build-core against the repo-root CMake (`GPC_BUILD_PYTHON`). Day-to-day façade work can still use `python/Makefile` + root `libgp`. See [`python/README.md`](../python/README.md).
- **Optional CTest (Phase C):** `cmake -B build -DGPC_BUILD_TESTS=ON -DGPC_USE_MATIO=ON -DCMAKE_PREFIX_PATH="…openblas;…libmatio" && cmake --build build && ctest --test-dir build` mirrors the makefile matio-gate binaries. **Actions still hard-gates on make**; CMake ctest is maintainer/packaging confidence only ([CIP-0020](../cip/cip0020.md) cutover decision).
- Legacy fragments (`make.cygwin`, `make.fc6`, `make.macports`, `make.ubuntu`, `make.atlas`) and the archived 2020 stub `CMakeLists.txt.archive-2020` live under [`make/legacy/`](../make/legacy/); they are not CI targets.
- Maintainer diagnostics / fixture stampers (`diag*`, `genTest*`) live under [`tools/`](../tools/); binaries still build to the repo root.
- Tutorial PNG plots used by the README live under [`docs/images/`](images/).
- **Linux default links OpenBLAS** (`-lopenblas`; install `libopenblas-dev`). `make RELEASE=1` is `-O2 -DNDEBUG` with `GPC_CHECKS=0` (CIP-0013 P1/P2). Checking / reference builds keep `GPC_CHECKS=1`.
- Blessed macOS path: Homebrew `gcc` / `gfortran` (newest `g++-N` / `gfortran-N`), OpenBLAS, optional `libmatio`. Same `RELEASE=1` switch.
- Language floor is **C++17** (`-std=c++17`). Dynamic exception specifications were removed so the pin is no longer C++14 ([CIP-0002](../cip/cip0002.md)).
- Timing harness: `make RELEASE=1 bench` then `./benchGPc --kind ftc|dtcvar|fitc …` (set `OPENBLAS_NUM_THREADS=1` for the q=2 protocol).
- GPy compare + LL gate: `OPENBLAS_NUM_THREADS=1 python bench/run_bench.py` (writes `bench.csv`; withholds timings when likelihoods disagree).

## Optional OpenMP (CIP-0025 Track 2)

Stationary workspace fills (`fillSymWorkspace` / `fillRectWorkspace`) accept an optional `#pragma omp parallel for` over disjoint row writes. The default build **does not** require OpenMP; CI and the REQ-0007 bench protocol remain single-threaded.

```bash
# Linux (libgomp ships with GCC):
make GPC_OPENMP=1 testKern && ./testKern

# macOS Homebrew GCC (gcc formula includes libgomp):
make GPC_OPENMP=1 testKern && ./testKern

# CMake:
cmake -B build -DGPC_OPENMP=ON && cmake --build build
```

**Nesting policy**: when `GPC_OPENMP=1`, set `OPENBLAS_NUM_THREADS=1` (or the vendor equivalent `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`) to avoid oversubscribing CPU cores. The CIP-0025 Track 1 GEMM distances delegate to multi-threaded BLAS; running both simultaneously without pinning can hurt performance:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$(nproc)   # or a measured value
./testKern
```

The REQ-0007 documented one-thread bench protocol (`OPENBLAS_NUM_THREADS=1`, OpenMP off) remains the acceptance bar. Multi-thread measurements (P7) are recorded in [CIP-0025](../cip/cip0025.md) (2026-11-18 table): BLAS threads help most on GEMM-distance workloads (\(q \ge 8\)); OpenMP alone is a small VarDTC win; nesting is not clearly better — prefer `OPENBLAS_NUM_THREADS=1` with OpenMP unless re-measured. Raw TSV: [`bench/p7_results/p7_large.tsv`](../bench/p7_results/p7_large.tsv).
- Windows CI waits on a maintained Windows make path; it is out of scope until that fragment exists.

## Reference verification without MATLAB

Full MATLAB/`mex` is not assumed in CI. Numeric `.mat` fixtures under `matfiles/` load via **libmatio** when built with `USE_MATIO=1` (`_NDLMATIO`).

- Gate: `make USE_MATIO=1 matio-tests` then `make run-matio-tests`.
- Optimiser unit gate (no `.mat` fixtures): `make testOptimisable && ./testOptimisable` — also run from `run-matio-tests` ([CIP-0010](../cip/cip0010.md)).
- `CGp` synthetic surface gate: `make testCGp && ./testCGp` — prediction, scale learning, `optimiseX`, StableFactor, expect-throw stubs ([CIP-0011](../cip/cip0011.md)); also in `run-matio-tests`.
- Object-level MATLAB I/O (kernel/noise/model structs) remains behind `_NDLMATLAB`.
- Retiring MATLAB bridges entirely is [REQ-0003](../requirements/req0003_matlab-bridge-retirement.md) (deferred).

## GitHub Actions

| Workflow | Role |
| --- | --- |
| **Compile** | Ubuntu + macOS hard gate: build `gp` / `ivm` / `gplvm` / `bgplvm`, help smoke |
| **Reference tests** | Ubuntu + macOS hard gate: matio suite |
| **Coverage** | Ubuntu only: `COVERAGE=1`, lcov, Codecov upload, **`CGp.cpp` ≥ 95% line gate** |

Coverage stays on Ubuntu (gcov/lcov path). `codecov.yml` ignores `test*.cpp` (and local fixture helpers) so patch coverage tracks library code. `testGp` exercises `CGp` approximation arms (FTC–PITC) under matio; `testCGp` fills prediction / throw / inducing gaps. Configure repository secret `CODECOV_TOKEN` for authenticated uploads.

### Regenerating the `CGp.cpp` miss list

```bash
make USE_MATIO=1 COVERAGE=1 matio-tests && make run-matio-tests
# On macOS Homebrew GCC, pass matching gcov: lcov --gcov-tool gcov-16 …
lcov --ignore-errors inconsistent,empty --capture --directory . -o coverage.info
lcov --ignore-errors unused --remove coverage.info '/usr/*' '*/test*.cpp' \
  '*/tools/diagDtcvar.cpp' '*/tools/genTestGpDtcvar.cpp' -o coverage.info
CGP_COVERAGE_THRESHOLD=95 ./scripts/check_cgp_coverage.sh coverage.info
```

Raise the gate toward 98% once remaining intentional misses (jitter warnings, rare equals arms, `_NDLMATLAB`) are classified in [CIP-0011](../cip/cip0011.md) Appendix A.

## Ordered categorical missing labels (macOS)

Missing ordered targets are stored as NaN. Casting to `int` before `isnan` is unsafe: on ARM64 `(int)NaN` is `0`, so missing points were scored as category 0. `COrderedNoise` tests `isnan` first. That was a genuine portability bug, not an OpenBLAS mismatch; after the fix, macOS Reference tests are a hard gate ([CIP-0006](../cip/cip0006.md), backlog `2026-11-13_macos-ordered-categorical-noise-mismatch`).
