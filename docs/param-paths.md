# GPy-like parameter paths (CIP-0014)

Path get/set is a façade over indexed `setParam` / `getParam`. Canonical leaf and component names match GPy/`paramz` where the parameterisation allows. Indexed APIs and stream I/O stay authoritative; the façade is not on the SCG hot path.

**Design**: [CIP-0014](../cip/cip0014.md) · **Requirement**: [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md)

## Kernel paths

| Path | Object | Notes |
|------|--------|-------|
| `lengthscale` / `variance` | atomic `CKern` | RBF/Exp convert `lengthscale` ↔ stored `inverseWidth` |
| `inv_lengthscale` / `inverseWidth` | atomic RBF/Exp | set/get in GPc storage units (no conversion) |
| `rbf.variance`, `rbf_1.…` | `CCmpndKern` | GPy component labels; `_k` for duplicates |
| `weight_variance` / `bias_variance` | MLP / Poly | GPy spelling; legacy camelCase aliases accepted |
| `power` | RatQuad | alias of stored `alpha` |

`paramPaths()` lists GPy canonical names. Ambiguous or missing paths throw `ndlexceptions::Error`.

Runtime constraints by path: `constrainPositive`, `constrainZeroOne`, `constrainNone` (via `replaceTransform` / clear).

## Model paths (`CGp`)

| Path | Notes |
|------|-------|
| `kern.…` or bare compound path | Delegates to `pkern` (optional `kern.` prefix) |
| `beta` | Sparse approximations only |
| `scale.j` | When output scales are learnt |

`fix` / `unfix` / `isFixed` use the same path grammar. Fix masks are not stream-persisted yet; loads default to all free.

## CLI

```bash
gp learn -k rbf -P rbf.variance=1.0 -P rbf.lengthscale=2.0 examples/sinc.svml sinc.model
```

`-P` / `--set path=value` is repeatable and applied after the model is built.

## Checks

```bash
make testParamPaths && ./testParamPaths
make USE_MATIO=1 testCGp && ./testCGp   # includes testFixMask / testModelPaths
```
