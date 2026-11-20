# SVI Bayesian GPLVM (uncollapsed, sampled \(q(\mathbf{X})\))

**Design:** [CIP-0030](../../cip/cip0030.md) · [REQ-0018](../../requirements/req0018_bayesian-gplvm-without-psi-statistics.md)  
**Collapsed sibling:** [bgplvm_oil100](./bgplvm_oil100.md) (`--inference collapsed`, default)

Use this path when psi statistics are unavailable (MLP / weight-space, white-noise, Green's inducing) or when \(N\) needs minibatches. Prefer collapsed `CBayesGplvm` when psi exist and the data fit in memory.

## Build

```sh
make bgplvm
```

## CLI tutorial

```sh
./bgplvm -v 2 -s 1 learn --inference svi -x 2 -M 10 -# 200 \
    --batch 50 --samples 2 --inducing points \
    --natgrad-gamma 0.1 --local-lr 1e-2 --adam-lr 1e-2 \
    -k rbfard -o oil100_svi \
    examples/oilTrain100.svml oil100_svi.model

./bgplvm display oil100_svi.model
./bgplvm gnuplot oil100_svi.model oil100_svi
```

Weight-space MLP (no psi):

```sh
./bgplvm -v 2 -s 1 learn --inference svi -x 2 -M 20 -# 300 \
    --inducing weightspace -k mlp \
    examples/oilTrain100.svml oil100_svi_mlp.model
# ARD on latent dims: -k mlpard   or   -k mlp -i true
```

### Flags (SVI)

| Flag | Role |
|------|------|
| `--inference svi` | Uncollapsed sampled \(q(\mathbf{X})\) on `CSvgp` + Adam |
| `--batch B` | Minibatch size (default \(N\)) |
| `--samples K` | Reparameterisation samples per point |
| `--inducing …` | `points` \| `whitenoise` \| `weightspace` \| `greens` |
| `--natgrad-gamma` | CVI natural-gradient step on \(q(\mathbf{v})\) in \([0,1]\) |
| `--local-lr` | Row-sparse Adam rate for \((\boldsymbol\mu,\log\mathbf S)\) |
| `--adam-lr` | Adam rate for kernel / inducing / \(\beta\) |

Stream type is `svibayesgplvm` (collapsed remains `bayesgplvm`). `-o` still writes `*_mu.dat` / `*_S.dat`.

## Notes

- Default `bgplvm learn` stays collapsed (CIP-0017).
- Python façade for `CSviBayesGplvm` is a follow-on backlog item; use the CLI or `testSviBayesGplvm` for now.
- MRD with a shared \(q(\mathbf{X})\) is deferred (CIP-0030 Phase 4 follow-on).
