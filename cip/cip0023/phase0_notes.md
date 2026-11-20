# CIP-0023 Phase 0 notes

Date: 2026-11-18

## Verdict: **continue revive**

Octave 10.3.0 can load Nat King’s PPA stack (`~/lawrennd/ppa` + kern/noise/ndlutil/optimi/prior/mltools/netlab) and run E-step + kernel update on a demPpa1 subset. Fixtures live under `testdata/ppa/`.

## How to regenerate

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/hdf5/1.14.5/lib   # Octave 10 vs hdf5 2.x mismatch on this machine
octave-cli --quiet scripts/dump_ppa_phase0_fixtures.m
```

## Fixture summary (`testdata/ppa/phase0_meta.txt`)

| Field | Value |
|-------|--------|
| Data | first 30 rows of `ppa/data/rand_{data,labels}.asc` |
| Noise | probit |
| Kernel | rbf + bias + white (cmpnd) |
| Options | `scalarB=1`, `varKern=0`, `updateBeta=0`, 3 outer iters, 10 kern iters |
| loglike_init | −31.740631380651045 |
| loglike_after | −24.346211953663285 |

Dumps (`.txt`, not `.dat` — repo ignores `*.dat`): `X`, `y`, `Kstore`/`invKstore`, expectations `f`/`ff`/`fBar`/`fBarfBar`, `g`/`nu`/`gamma`, `B`, kern params — at **init** (after `ppaInit`) and **after** short EM.

## Upstream quirks

1. **`ppaMStep.m` is syntactically incomplete** (missing `end` tokens). Octave refuses to parse it.
2. **`ppaOptimiseKernel.m` returns immediately** (extra `end` after the `nargin` block), so kernel M-step is a no-op in the shipped toolbox. Phase 0 **after** fixtures are therefore **E-step-only** (`kernIters=0`). C++ kernel M-step is checked with `checkGradients`, not those tensors.
3. **`ppa.m` omits semicolons** on `model.X = X` / `model.y = y` (noisy stdout).
4. On this host, Homebrew Octave needs the **hdf5 1.14.5** library path (built against `libhdf5.310`, while `hdf5` formula is 2.2 / `.320`).

## Design locks from Phase 0

- Full-data \(N\times N\) `Kstore` / `invKstore` (not IVM active-set).
- First ship: **scalar β**, **non-variational** kernel M-step, probit + rbf/bias/white.
- Drop live dependence on broken `ppaMStep.m`; port M-step logic into C++ from `ppaUpdateKernel` / `ppaOptimiseKernel` instead.
