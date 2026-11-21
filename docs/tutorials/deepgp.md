# Deep Gaussian processes (doubly stochastic VI)

**Design:** [CIP-0035](../../cip/cip0035.md) · [REQ-001A](../../requirements/req001A_deep-gaussian-processes.md)  
**Core:** `CDeepGp` stacks `CSvgp` layers with fixed mean maps and minibatch Adam.

Use this path for compositional regression when a shallow sparse GP underfits. Prefer a single `CSvgp` / `./gp` when one layer is enough.

## Build

```sh
make deepgp testDeepGp
```

## CLI tutorial

```sh
./deepgp -v 2 -s 1 learn -L 2 --hidden 1 -M 10 -# 200 \
    --batch 50 --samples 1 --inducing points --chi 1.0 \
    --adam-lr 1e-2 -k rbf \
    examples/sinc.svml sinc_deep.model

./deepgp display sinc_deep.model
```

Weight-space ridges (learned window width via the library API; CLI uses the layer default):

```sh
./deepgp -v 2 -s 1 learn -L 2 -M 16 -# 300 \
    --inducing weightspace -k mlp \
    examples/sinc.svml sinc_ws.model
```

Finite-network mode (Nyström residual off — a bound on a different model):

```sh
./deepgp -v 2 -s 1 learn -L 2 -M 16 -# 200 \
    --inducing weightspace --residual off -k mlp \
    examples/sinc.svml sinc_finite.model
```

### Flags

| Flag | Role |
|------|------|
| `-L N` | Number of GP layers (default 2) |
| `--hidden D` | Width of intermediate representations (default \(D_\mathrm{in}\)) |
| `-M M` or `-M M1,M2,…` | Inducing count per layer |
| `--inducing …` | `points` \| `weightspace` |
| `--residual on\|off` | Keep (default) or drop Nyström residual \(k-\mathbf a^\top\mathbf a\) |
| `--batch B` | Minibatch size (default \(N\)) |
| `--samples K` | Reparameterisation samples per point |
| `--chi c` | Hidden-layer variance scale (\(c/(L-1)\) on hidden kernels) |
| `--adam-lr` | Adam step size |
| `--natgrad-gamma` | Optional final-layer CVI step in \([0,1]\) |

Stream type is `deepgp`.

## Residual-off

With `--residual off`, each layer’s predictive variance omits the Nyström leftover. That is the finite Bayesian network bound discussed in CIP-0035 (prototype weight-space stacks), not the same model as residual-on sparse deep GPs. Prefer residual-on with inducing points for the standard deep GP; use residual-off when comparing to finite networks or diagnosing window underfit.

## Jacobian diagnostic (V7)

`CDeepGp::jacobianSpread` reports the median log singular-value spread of the prior map under identity mean maps. With \(\chi=1/L\) on every layer the spread stays \(O(1)\) in depth (CIP-0035 table; `testDeepGp` Phase 4 V7).

## Python façade

```python
import gpc
m = gpc.models.DeepGP(X, Y, layers=2, hidden=1, num_inducing=10, chi=0.5)
m.optimize(iters=100, adam_lr=1e-2, seed=1)
mu, var = m.predict(X)
m.save("sinc_deep.model")
```

See [`python/examples/deepgp_sinc.py`](../../python/examples/deepgp_sinc.py) and `python/tests/test_deepgp.py`. Probit / ordered finals: `likelihood='probit'` or `'ordered'`.

## Notes

- Non-Gaussian finals on the CLI still need a follow-on; use the Python façade or `CDeepGp::setLikelihood` in C++.
