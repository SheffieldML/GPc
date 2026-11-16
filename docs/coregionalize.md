# Multi-output coregionalization (CIP-000D)

GPc follows GPy’s ICM/LCM construction.

## Data layout (`build_XY`)

Stack one block per output. Append the output index as the **last** column of `X`, stored as a `double` holding `0, 1, …, D-1`. `y` is the same vertical stack (one column).

Example for two outputs with one spatial input:

| x | index |
|---|-------|
| … | 0.0   |
| … | 0.0   |
| … | 1.0   |

## Kernels

- `coregionalize` — \(B = WW^\top + \mathrm{diag}(\kappa)\), reads the index column via `activeDims`
- `tensor` of spatial × `coregionalize` — ICM (`createIcmKern`)
- sum of ICMs — LCM (`createLcmKern`)
- `indexwhite` — per-output noise diagonal; **sum outside** the tensor, never inside it

Index values are cast with `llround`, must be within `MATCHTOL` of an integer, and in `0 … D-1`.

## Checks

```bash
make testCoregionalize && ./testCoregionalize
```
