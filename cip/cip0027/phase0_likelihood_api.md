# CIP-0027 Phase 0 — Likelihood API sketch

Maps existing `CNoise` surfaces onto the [target architecture](../../docs/target-architecture.md) rule-2 names. Implementation: free functions in `ndlnoise` (`CNoise.h` / `CNoise.cpp`). **No new fields** on `CNoise` subclasses.

| Architecture / GPy2 name | Batched helper (`ndlnoise`) | Existing oracle / wrapper |
|---|---|---|
| `log_density` | `gaussianLogDensity` | (none dedicated; VE at \(v=0\)) |
| `variational_expectations` | `gaussianVariationalExpectations` | `CGaussianNoise::expectedLogLikelihood` |
| `log_predictive_density` | `gaussianLogPredictiveDensity` | `CGaussianNoise::logLikelihood(mu,var,y)` |
| `predictive_moments` | (Phase A+) | `out()` |
| `ep_moments` | deferred → [CIP-0033](../cip0033.md) | follow-on likelihood / inference |

**Gram helper for Phase A FTC:** `ndlnoise::addObservationNoiseDiag(K, sigma2)`.

**Parameter accessors (canonical name):** `CGaussianNoise::getVariance` / `setVariance` (aliases of `getSigma2` / `setSigma2`). Path wiring `likelihood.variance` / alias `Gaussian_noise.variance` is Phase A (CIP-0014).

**Out of Phase 0:** migrating IVM / PPA / ADF callers; stripping `py` / `mu` / `varSigma` from `CNoise`.
