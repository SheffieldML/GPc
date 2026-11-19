# CIP-0028 prototype (NumPy)

NumPy evidence for the 2026-11-19 ridge-window revision of [CIP-0028](../../cip0028.md).

Source archive: `deepws_experiment.zip` (unpacked as `deepws/`).

## Contents (`deepws/`)

| Path | Role |
|---|---|
| `wsfeatures.py`, `ridge.py` | Isotropic √p and ridge closed forms |
| `check_features.py` | Quadrature / Monte Carlo checks |
| `dim_sweep.py`, `ridge_sweep.py`, `budget_sweep.py` | Variance-captured tables under \(\kappa(K_{uu})\le10^4\) |
| `deep.py`, `ws_deep.py`, `tied_*.py` | Jacobian-spread / deep design notes (non-goal of this CIP) |
| `*.json`, `deep_ws_jacobians.png` | Cached results and figure |
| `cip0028_update.md` | Draft section appended to the CIP |

## Reproduce

```bash
cd deepws
python check_features.py
python dim_sweep.py
python ridge_sweep.py
python budget_sweep.py
```

Deep Jacobian sweeps need the high-precision stack used when the JSON was produced; prefer the checked-in `*_results.json` / `*_sweep.json` unless re-running intentionally.
