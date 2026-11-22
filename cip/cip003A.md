---
author: "Neil Lawrence"
created: "2026-11-22"
id: "003A"
last_updated: "2026-11-22"  # Phase 5 done → Implemented
status: "Implemented"
compressed: false
related_requirements: ["001B", "0014", "0019", "000E", "0002", "0017"]
related_cips: ["0027", "0033", "0034", "0036", "001D", "002A", "0035", "0015"]
tags:
- cip
- likelihood
- noise
- gpy
- python
- facade
- skew
- student-t
- poisson
- parity
title: "GPy-aligned likelihood zoo in CNoise and gpc.likelihoods"
---

# CIP-003A: GPy-aligned likelihood zoo in `CNoise` and `gpc.likelihoods`

> CIPs describe HOW to achieve requirements (WHAT).  
> Implements [REQ-001B](../requirements/req001B_gpy-aligned-likelihood-zoo.md).  
> Architecture: [docs/target-architecture.md](../docs/target-architecture.md) · likelihood API: [cip0027/phase0_likelihood_api.md](./cip0027/phase0_likelihood_api.md).  
> Skew design mirror: GPy CIP-0014 (`GPy/likelihoods/skewed.py`, `GPy/util/gaussian_special.py`).  
> Worktree: `GPc-cip003A` / branch `cip003A-likelihood-zoo`.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Close the documented **broader likelihood zoo** gap: port the GPy observation families GPc still lacks into `CNoise` / `ndlnoise`, register them for stream I/O, drive them through existing Laplace / EP / SVGP hooks, and expose a GPy-facing **`gpc.likelihoods`** module plus model wiring so notebooks can construct `SkewGaussian`, `StudentT`, `Poisson`, etc. without writing C++.

**Which requirements?** [REQ-001B](../requirements/req001B_gpy-aligned-likelihood-zoo.md) (primary); [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md), [REQ-0019](../requirements/req0019_nonconjugate-inference-ep-laplace.md), [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md), [REQ-0002](../requirements/req0002_runnable-verification.md), [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md).

## Motivation

GPc today ships five noise types — `gaussian`, `scale`, `probit`, `ncnm`, `ordered`. GPy exports Gaussian, HeteroscedasticGaussian, Bernoulli, Binomial, Poisson, Exponential, Gamma, StudentT, Weibull, LogLogistic, SkewGaussian, SkewExponential, MixedNoise, MultioutputLikelihood (plus unexported LogGaussian). Tutorial CIPs and REQ-000E explicitly deferred Poisson / Student-t / hetero / warped from the **first** subset; [docs/target-architecture.md](../docs/target-architecture.md) still lists the zoo as open.

Two pressures make this the right next HOW:

1. **Parity.** Teachers and GPy users now hit a hard stop when a notebook needs robust or count observations.
2. **Skew.** GPy CIP-0014 (2026-11-22) restored O'Hagan / Azzalini skew likelihoods for Ferrari-style lap-time modelling. GPc should own the same families in C++ so both engines can share fixtures and gradients.

## Detailed Description

### Capability matrix (GPy → GPc)

Status key: **Match** · **Gap** (this CIP) · **Defer** · **GPc-only**.

| GPy / family | GPc today | Status | Inference note |
|---|---|---|---|
| Gaussian | `CGaussianNoise` | **Match** | Exact / FTC |
| Bernoulli + Probit | `CProbitNoise` | **Match** | EP / Laplace (CIP-0033) |
| Ordinal | `COrderedNoise` | **GPc-only** | Keep; not in current GPy tree |
| NCNM | `CNcnmNoise` | **GPc-only** | Keep |
| Scale (GPLVM) | `CScaleNoise` | **GPc-only** | Not HeteroscedasticGaussian |
| StudentT | `CStudentTNoise` | **Match** | Laplace + GH VE |
| Poisson | `CPoissonNoise` | **Match** | Laplace + GH VE; log link |
| Binomial | `CBinomialNoise` | **Match** | Laplace + GH VE; probit link |
| Exponential | `CExponentialNoise` | **Match** | Laplace + GH VE |
| Gamma | `CGammaNoise` | **Match** | Laplace + GH VE |
| SkewGaussian | `CSkewGaussianNoise` | **Match** | Laplace + GH VE |
| SkewExponential | `CSkewExponentialNoise` | **Match** | Laplace + GH VE |
| HeteroscedasticGaussian | — | **Defer** | Needs per-point variance channel |
| Weibull / LogLogistic / LogGaussian | — | **Defer** | Survival / censoring metadata |
| MixedNoise / MultioutputLikelihood | — | **Defer** | Container types; coregionalise kernels already exist |
| Warped GP | — | **Defer** | Separate CIP |

### Design principles

1. **Compose through `CNoise`.** Each family is a subclass (or a thin typed wrapper over `ndlnoise` free functions). No special-casing inside `CGp` beyond the existing `isGaussianNoise` / inference-engine dispatch.
2. **Batched pure oracles in `ndlnoise`.** Prefer free functions for log-density, \(d\log p/df\), \(d^2\), \(d^3\), hyperparameter derivatives, and (where closed form exists) variational expectations — same split as CIP-0027 Phase 0 / CIP-0033 probit helpers.
3. **Inference reuse.** Non-conjugate families ride **Laplace** (and EP quadrature where GPy falls back to the base `Likelihood.moments_match_ep`). Exact FTC remains Gaussian-only. SVGP / DeepGP get `expectedLogLikelihood` when a closed form or stable GH quadrature exists; otherwise document "Laplace-only in v1".
4. **Skew = GPy CIP-0014.**  
   \(p(e)=2\,g(e)\,G(\alpha e)\) with Gaussian or bilateral-exponential base. Shared **inverse Mills ratio** \(\varphi/\Phi\) and first two derivatives in C++ (mirror `GPy.util.gaussian_special`). `alpha=0` on SkewGaussian must match `CGaussianNoise` within tolerance.
5. **Python façade, not a second engine.** `gpc.likelihoods` constructs / configures C++ objects; models accept `likelihood=` as an object or a documented string. Parameter paths stay CIP-0014 style (`likelihood.variance`, `likelihood.alpha`, `likelihood.deg_free`, …).
6. **Verify against GPy / SciPy.** Generator scripts under `cip/cip003A/` emit fixtures (densities, grads, small Laplace fits). C++ `checkGradients` + Python tests gate merge.

### C++ surface (sketch)

| Class | Stream `type` | Key params (canonical paths) | Log-concave? |
|---|---|---|---|
| `CStudentTNoise` | `student_t` | `likelihood.variance`, `likelihood.deg_free` (ν) | no |
| `CPoissonNoise` | `poisson` | (none, or rate link params if any) | yes* |
| `CBinomialNoise` | `binomial` | `likelihood.num_trials` (fixed / data) | yes* |
| `CExponentialNoise` | `exponential` | — | yes* |
| `CGammaNoise` | `gamma` | `likelihood.beta` (or shape) | yes* |
| `CSkewGaussianNoise` | `skew_gaussian` | `likelihood.variance`, `likelihood.alpha` | no |
| `CSkewExponentialNoise` | `skew_exponential` | `likelihood.rate`, `likelihood.alpha` | no |

\*With the usual log / probit / identity link choices GPy uses; mark `isLogConcave` consistently with GPy's `log_concave` flag.

Shared helpers (new header, e.g. `ndlnoise_special.h` or under `ndlutil`):

- `invMillsRatio`, `dInvMillsRatio`, `d2InvMillsRatio`
- `logPhi` / `logNdtr` via `erf` / `erfc` (already used in MLP ridge features)

Extend `readNoiseFromStream` if-chain; bump stream field docs only if parameter layouts need a version bump (prefer additive fields under `NOISEVERSION` rules already used by ordered / ncnm).

### Python façade (sketch)

```python
import gpc

lik = gpc.likelihoods.SkewGaussian(variance=0.1, alpha=2.0)
# or StudentT, Poisson, Binomial, Exponential, Gamma, SkewExponential,
#    Gaussian, Bernoulli (aliases of existing noise)

m = gpc.models.GPRegression(X, Y, kernel=k, likelihood=lik, inference="laplace")
m.optimize()
mu, var = m.predict_noiseless(Xtest)
```

- New module `python/gpc/likelihoods.py` + pybind registrations in `python/src/`.
- Keep string shortcuts (`likelihood="gaussian"`) working; document which strings map to which classes.
- `GPClassification` stays Bernoulli/probit-centric; Poisson / Student-t / skew attach to regression-style constructors with an explicit `inference=` (default `laplace` when non-Gaussian).
- DeepGP / SVGP: accept the new types only when `expectedLogLikelihood` is implemented; else raise a clear error.

### Phasing

| Phase | Deliverable |
|---|---|
| **0** | Inventory + fixture generators; shared Mills / Φ helpers; capability matrix frozen in this CIP |
| **1** | SkewGaussian + SkewExponential (`CNoise` + `ndlnoise` + stream + `checkGradients` + SciPy/GPy fixtures) |
| **2** | StudentT + Poisson (+ Binomial if cheap) with Laplace on `CGp` |
| **3** | Exponential + Gamma |
| **4** | `gpc.likelihoods` + model `likelihood=` wiring + docstring/help (REQ-0017) |
| **5** | SVGP / DeepGP VE hooks where closed form or GH exists; docs + subset checklist update |

Phases 1–3 may land as one PR series if review prefers; Phase 4 must not precede at least one non-Gaussian family beyond probit.

### Alternatives considered

- **Python-only likelihoods calling into a generic quadrature C++ GP** — rejects Flexibility Without a Performance Tax and Compose Through Interfaces.
- **Single mega-`CGenericNoise` with function pointers** — harder to stream, name, and gradient-check; keep typed subclasses.
- **Defer skew until survival likelihoods** — skew is already in GPy and has a clear Ferrari use case; pull it forward in Phase 1.

## Implementation Plan

1. **Phase 0 — Foundations**
   - Add `cip/cip003A/` generators mirroring GPy `test_skewed_likelihood.py` and Student-t / Poisson density checks
   - Implement shared Gaussian-tail helpers; unit-test vs SciPy
   - Confirm Laplace / EP call sites only need the existing `CNoise` virtuals + `ndlnoise` batch helpers

2. **Phase 1 — Skew**
   - `CSkewGaussianNoise`, `CSkewExponentialNoise`
   - Register stream types; path names `likelihood.variance` / `rate` / `alpha`
   - Reference: SciPy `skewnorm.logpdf`; `alpha=0` ≡ Gaussian

3. **Phase 2 — Robust and count**
   - `CStudentTNoise`, `CPoissonNoise`, `CBinomialNoise`
   - Wire Laplace on `CGp` (reuse CIP-0033 engine); EP via quadrature unless closed form is short

4. **Phase 3 — Exponential family remainder**
   - `CExponentialNoise`, `CGammaNoise`

5. **Phase 4 — Python façade**
   - `gpc.likelihoods` module; pybind constructors
   - `GPRegression(..., likelihood=..., inference=...)` (and sparse / SVGP as supported)
   - Tests under `python/tests/`; update `python/docs/subset_checklist.md`

6. **Phase 5 — Variational hooks + docs**
   - `expectedLogLikelihood` / GH where needed for SVGP / DeepGP
   - Tutorial note + compress checklist for REQ-001B

## Backward Compatibility

- Existing models and stream files unchanged; new `type` strings only.
- Default `GPRegression` remains Gaussian + exact inference.
- Former non-goal wording in REQ-000E / CIP-0015 stays historically accurate for the *first* subset; REQ-001B owns the expansion.
- CLI (`gp`, `ivm`, …): optional later backlog to expose new types; not a Phase 1–4 gate.

## Testing Strategy

- C++: `checkGradients` per new `CNoise`; density / derivative fixtures from GPy or SciPy; `alpha=0` skew ≡ gaussian; Student-t large-ν ≈ gaussian smoke
- Python: construct each `gpc.likelihoods.*`, optimise a tiny seeded problem under Laplace, compare objective / params to GPy within tolerance
- No MATLAB `.mat` dependency for new families (matio path remains for legacy noise tests only)
- Assert dimension / bound checks at API boundaries (project tenet)

## Related Requirements

- [REQ-001B](../requirements/req001B_gpy-aligned-likelihood-zoo.md) — WHAT this CIP implements
- [REQ-0014](../requirements/req0014_observation-noise-in-likelihood.md) — likelihood-owned noise parameters
- [REQ-0019](../requirements/req0019_nonconjugate-inference-ep-laplace.md) — EP / Laplace over pure likelihoods
- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) — Python ergonomics (extended beyond original non-goals)
- [REQ-0002](../requirements/req0002_runnable-verification.md), [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md)

## Implementation Status

- [x] Draft CIP + REQ-001B; capability matrix
- [x] Accepted; worktree `cip003A-likelihood-zoo`
- [x] Phase 0 helpers + fixture generators (`ndlnoise::invMillsRatio*`, `cip/cip003A/gen_skew_fixture.py`)
- [x] Phase 1 skew likelihoods (`CSkewGaussianNoise`, `CSkewExponentialNoise`, `./testSkewNoise` green)
- [x] Phase 2 Student-t / Poisson / Binomial (`./testLikelihoodZoo` green)
- [x] Phase 3 Exponential / Gamma (`CExponentialNoise`, `CGammaNoise`)
- [x] Phase 4 `gpc.likelihoods` + DeepGP string wiring + `noise_log_density`
- [x] Phase 5 GH VE checks, generalised `CLaplaceInference`, `GPRegression(likelihood=, inference='laplace')`, docs

## References

- GPy `GPy/likelihoods/` (`student_t.py`, `poisson.py`, `skewed.py`, …); GPy CIP-0014
- [CIP-0027](./cip0027.md), [CIP-0033](./cip0033.md), [CIP-0034](./cip0034.md), [CIP-0036](./cip0036.md)
- O'Hagan & Leonard (1976); Azzalini (1985)
- SciPy `stats.skewnorm`, `special.log_ndtr`
