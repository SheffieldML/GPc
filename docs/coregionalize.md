# Multi-output coregionalization (CIP-000D / CIP-0015)

GPc follows GPy’s ICM/LCM construction.

## Data layout (`build_XY`)

Stack one block per output. Append the output index as the **last** column of `X`, stored as a `double` holding `0, 1, …, D-1`. `y` is the same vertical stack (one column).

Helpers in [`CMultioutput.h`](../CMultioutput.h) (CIP-0015 Phase A):

| Helper | Role |
|--------|------|
| `buildXY(Xlist, Ylist, Xout, Yout)` | GPy `build_XY` stacking |
| `getSlices` | Contiguous per-output row ranges (GPy `get_slices`) |
| `indexToSlices` | Runs of equal indices (GPy `index_to_slices`) |
| `stampOutputIndex(X, d)` | Set last column of `X*` to output `d` for prediction |

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
- `createCoregionalizedKern` / `createCoregionalizedLcmKern` — ICM or LCM plus optional `indexwhite` as a `CCmpndKern`

Index values are cast with `llround`, must be within `MATCHTOL` of an integer, and in `0 … D-1`.

## Trainable model factory

`CCoregionalizedRegression` owns stacked `X`/`y`, the compound kernel, a spherical `CGaussianNoise` (likelihood variance fixed small), and an FTC `CGp`. Per-output noise lives in `indexwhite` (GPy `MixedNoise` spirit).

```cpp
CRbfKern spatial(1);
CCoregionalizedRegression model(Xlist, Ylist, spatial);
model.getModel().optimise();
CMatrix Xstar(/* … */);
stampOutputIndex(Xstar, /* output */ 0);
model.getModel().posteriorMeanVar(mu, var, Xstar);
```

## Demo

```bash
make demo_coregionalization_toy && ./demo_coregionalization_toy
```

Tutorial: [docs/tutorials/coregionalization_toy.md](./tutorials/coregionalization_toy.md).

## Checks

```bash
make testCoregionalize && ./testCoregionalize
```
