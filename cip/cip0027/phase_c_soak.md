# CIP-0027 Phase C — Harden + algorithm performance soak

Date: 2026-11-20.  
Binary: `./testCip0027Perf` (`testCip0027Perf.cpp`).  
Also: CLI `--allow-white-and-noise` on `./gp learn`.

## Policy (Gaussian FTC regression)

| Situation | Behaviour |
|-----------|-----------|
| Trainable White **and** free `likelihood.variance` | **Throws** on `optimise` / `logLikelihoodGradient` |
| Legacy White-owned noise | `fixLikelihoodVariance(true)` (CLI auto-fixes when `-k white` without allow) |
| Intentional dual noise | `setAllowWhiteAndLikelihoodNoise(true)` / `--allow-white-and-noise` |

Dual free noise is no longer a silent training path. IVM is unaffected (probit + White softener; no Gaussian `likelihood.variance` dual).

## Algorithm performance gates (regression)

| Gate | Protocol | Floor / ceiling (soak) |
|------|----------|------------------------|
| FTC sinc LL | `examples/sinc.csv`, hold out 8, SCG 40 | trained LL ≥ 15; LL improves by ≥ 1 |
| FTC sinc RMSE | posterior mean on hold-out | RMSE ≤ 0.25 |
| FTC sinc σ² | after train | `likelihood.variance` ≤ 0.05 |
| FTC ≡ White | copy RBF + set `white.variance=σ²`, fix lik.var | LL / RMSE match within `MATCHTOL` |
| Sparse DTC | same split, M=8 inducing fixed, SCG 40 | LL ≥ 10; RMSE ≤ 0.5; β = 1/σ² |
| IVM softener (short) | `unitsquaregp.svml`, M=50, `optimise(3,40,15)` | RBF-only Jaccard ≤ 0.50; RBF+White ≥ 0.70 |
| IVM softener (deep) | `testIvmSoftener`: M=80, `optimise(4,50,15)` + unit gates | RBF-only Jaccard ≤ 0.35; RBF+White ≥ 0.80; |ΔLL| RBF-only > 0.5; White changes `selectPoints`; probit σ² fixed |

Observed on this machine (2026-11-20): FTC LL ≈ 23.5, RMSE ≈ 0.073, σ² ≈ 0.002; DTC LL ≈ 19.6; IVM short Jaccard RBF-only ≈ 0.32 / RBF+White ≈ 0.92; long ≈ 0.09 / 1.00.

## Decision

Keep the harden (refuse dual free noise by default). Do **not** drop the allow escape hatch — intentional White structure plus likelihood noise remains available under the flag. Dual-path Gram skip for fixed-variance + White stays for legacy models.

## Reproduce

```sh
make testCip0027Perf
./testCip0027Perf
```
