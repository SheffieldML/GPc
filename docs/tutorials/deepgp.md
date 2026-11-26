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
| `--inducing …` | `points` \| `weightspace` \| `hermite` (needs `-k mlp`) |
| `--residual on\|off` | Keep (default) or drop Nyström residual \(k-\mathbf a^\top\mathbf a\) |
| `--batch B` | Minibatch size (default \(N\)) |
| `--samples K` | Reparameterisation samples per point |
| `--chi c` | Hidden-layer variance scale (\(c/(L-1)\) on hidden kernels) |
| `--adam-lr` | Adam step size |
| `--natgrad-gamma` | Optional final-layer CVI step in \([0,1]\) |

Stream type is `deepgp`.

## Hermite windows (`inducing='hermite'`)

`inducing='hermite'` replaces the inducing points with a fixed family of Hermite ridge windows, \(\psi_{v,k}(\omega)=\mathrm{He}_k(v\cdot\omega)/\sqrt{k!}\,\sqrt{N(\omega)}\) with odd degrees \(k\le 9\) (CIP-003D). It needs `kernel='mlp'`.

What changes compared with Gaussian ridges (`inducing='weightspace'`):

- *No smoothed-erf reading.* A Gaussian window has a width \(s\) and an offset, and can be read as a smoothed erf unit. A Hermite window has neither. The family is a fixed orthogonal basis for the MLP kernel's weight-space expansion, not a set of units to inspect.
- *A well-conditioned \(K_{uu}\).* \(K_{uu}[m,n]=\delta_{k_mk_n}(v_m\cdot v_n)^{k}\) does not involve the kernel, so it is block diagonal by degree. Gaussian windows at overlapping widths give a \(K_{uu}\) that needs jitter.
- *Nothing to optimise.* The windows have no free parameters. Only the kernel parameters, the noise and (for a deep GP) the variational parameters move.
- *A capacity.* With \(D'=d+1\), degree 1 holds \(D'\) windows and each odd degree \(k\ge 3\) holds \(\lfloor\binom{D'+k-1}{k}/2\rfloor\):

| input dimension \(d\) | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| maximum \(M\) | 16 | 63 | 212 | 1557 | 16106 |

### What you are told when the request cannot be delivered

Each case below either raises or warns, so the layout you asked for is never silently replaced by another.

| Code | Condition | Outcome |
|---|---|---|
| E1 | `num_inducing` (or the rows of `Z`) exceeds the capacity | raises, naming both numbers |
| E2 | the kernel is not `mlp` | raises |
| E3 | `window_width` or `learn_window_width(s)` given | raises (the windows have no width) |
| E4 | `inducing_fixed=False` given to `SparseGPRegression` | raises |
| W1 | \(\kappa(K_{uu})>10^4\) | `UserWarning` |
| W2 | more than 90% of capacity used | `UserWarning` |
| W3 | captured-variance fraction below 0.5 | `UserWarning` (see the caveat below) |
| W4 | Hermite in a hidden layer of a multi-layer deep GP | `UserWarning` |

`SparseGPRegression` leaves `inducing_fixed` at `None`, which resolves to fixed for Hermite and to `False` for the other families. `str(model)` always shows the delivered layout (degree allocation, fill, \(\kappa\), captured variance), whether or not a warning fired.

**Hidden layers.** W4 is a caution, not a prohibition: the CIP-0035 comparison that motivates the family was made on the final layer. A hidden layer sees an input distribution that changes during training, and fixed windows cannot follow it.

**Caveat on W3.** The captured-variance fraction compares the Nyström diagonal with the prior diagonal on the training inputs. For the MLP kernel it settles between 0.67 and 0.83 in every case we tried (\(d\) from 1 to 5, input scales from 0.1 to \(10^5\)), because degrees above 9 are absent from the family. The 0.5 threshold is a placeholder that those cases never reach, so W3 is covered by a C++ test with an explicit threshold and not by a Python test. Phase 6 of CIP-003D calibrates it.

```python
m = gpc.models.DeepGP(X, Y, layers=1, num_inducing=64, kernel="mlp", inducing="hermite")
print(m)   # shows the layout and the fixed-by-construction line
```

## Residual-off

With `--residual off`, each layer’s predictive variance omits the Nyström leftover. That is the finite Bayesian network bound discussed in CIP-0035 (prototype weight-space stacks), not the same model as residual-on sparse deep GPs. Prefer residual-on with inducing points for the standard deep GP; use residual-off when comparing to finite networks or diagnosing window underfit.

## Jacobian diagnostic (V7)

`CDeepGp::jacobianSpread` reports the median log singular-value spread of the prior map under identity mean maps. Rectangular maps (\(D_\mathrm{in}\neq D_\mathrm{out}\)) are allowed; the spread is over the \(\min(D_\mathrm{in},D_\mathrm{out})\) singular values. With \(\chi=1/L\) on every layer the spread stays \(O(1)\) in depth (CIP-0035 table; `testDeepGp` Phase 4 V7).

## Python façade

```python
import gpc
m = gpc.models.DeepGP(X, Y, layers=2, hidden=1, num_inducing=10, chi=0.5)
m.optimize(iters=100, adam_lr=1e-2, seed=1)
mu, var = m.predict(X)
m.save("sinc_deep.model")
```

`predict` returns the variance of a new observation (for a Gaussian likelihood it includes \(1/\beta\)); `predict_noiseless` returns the latent-function variance. `hidden` may exceed the input dimension: the PCA mean path uses every input direction and zero-pads the extra hidden units, which the GP layer alone drives. Successive `optimize` calls continue the same Adam state (`reset_optimizer()` restarts it), so time-to-target curves can be recorded in chunks. For weight-space layers, `learn_window_width=True` (tied) or `learn_window_widths=True` (per-unit) learns the window widths.

See [`python/examples/deepgp_sinc.py`](../../python/examples/deepgp_sinc.py) and `python/tests/test_deepgp.py`. Probit / ordered finals: `likelihood='probit'` or `'ordered'`.

## Notes

- Non-Gaussian finals on the CLI still need a follow-on; use the Python façade or `CDeepGp::setLikelihood` in C++.
