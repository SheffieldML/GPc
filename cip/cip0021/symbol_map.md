# MATLAB → C++ symbol map (CIP-0021)

Reference root: `~/lawrennd/ppa/matlab` (GitHub [lawrennd/ppa](https://github.com/lawrennd/ppa)).  
GPc targets: `CPpa.{h,cpp}`, `ppa.h` (restore), `ppa.cpp`.

| MATLAB | Role | C++ target |
|--------|------|------------|
| `ppa.m` | Construct model, noise, kern, expectation storage | `CPpa` ctor |
| `ppaOptions.m` | Outer/kern/β iters, `scalarB`, `varKern`, tol | ctor args / setters (mirror IVM optimiser budget style) |
| `ppaInit.m` | `Kstore`, `invKstore`, init `B`, zeros, first E-step | `init` / `initStoreage` / `initVals` |
| `ppaOptimisePPA.m` | Outer EM loop | `optimise` |
| `ppaEStep.m` | f̄ moments → ν,g,γ → f,ff | `eStep` |
| `ppaExpectfBar.m` | \(\langle\bar f\rangle\) | `updateExpectationsfBar` (implement; currently throws) |
| `ppaExpectfBarfBar.m` | \(\langle\bar f\bar f^\top\rangle\) | same / dedicated helper; store as `C` or `fBarfBar` |
| `ppaExpectf.m` | \(\langle f\rangle\) | `updateExpectationf` (present; verify vs fixture) |
| `ppaExpectff.m` | \(\langle f f\rangle\) | `updateExpectationff` (present; verify) |
| `noiseUpdateNuG` | site params | `CNoise::getNuG` via `updateNuG` |
| `ppaMStep.m` | kern (+ optional β) | `mStep` — call kernel optimiser; **do not parse broken upstream file** |
| `ppaUpdateKernel.m` | `ppaOptimiseKernel` then recompute K, invK | `updateK` + `runDefaultOptimiser` on kern params |
| `ppaOptimiseKernel.m` / `ppaKernel*` | SCG on kern | `COptimisable` / `approxLogLikelihoodGradient` |
| `ppaScgUpdateBeta.m` / `ppaUpdateB*` | β M-step | Phase 2+ (`updateBeta` option) |
| `ppaCalculateLogLike.m` | training approx LL | `approxLogLikelihood` / `logLikelihood` |
| `ppaCovarianceGradient.m` | dLL/dK | `updateCovGradient` |
| `ppaPosteriorMeanVar.m` | predictive | `posteriorMeanVar` |
| `ppaOut.m` | predictions | `out` |
| `demPpa1.m`–`4` | demos | CLI `./ppa learn` + tutorials after Closed |
| `ppa/src/CPpa.*` | unfinished C++ | **ignore** — same stubs as GPc |

## Storage mapping

| MATLAB field | C++ |
|--------------|-----|
| `model.X`, `model.y` | `X`, `y` (or pointers like IVM) |
| `model.B` | `beta` (scalar: `1× outputDim`; clarify vs IVM site β) |
| `model.kern.Kstore` | `K` |
| `model.kern.invKstore` | `invK` (+ `logDetK`) |
| `model.expectations.f` | `f` |
| `model.expectations.ff` | `ff` |
| `model.expectations.fBar` | `fBar` |
| `model.expectations.fBarfBar` | new `fBarfBar` or reuse `C[]` carefully |
| `model.g`, `model.nu`, `model.gamma` | `g`, `nu`, `gamma` |
| `model.noise` | `CNoise` |
| `model.kern` | `CKern` |

## Deliberate non-maps

- IVM `activeSet` / `inactiveSet` / entropy selection — **not** part of MATLAB PPA; remove from live C++ path.
- Mex `toMxArray` / `fromMxArray` — optional later; fixtures are ASCII for Phase 3.
