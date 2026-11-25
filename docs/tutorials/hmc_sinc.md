# Posterior samples of hyperparameters (sinc)

**GPy counterpart:** `m.optimize()` followed by `GPy.inference.mcmc.HMC` over the model parameters  
**GPc path:** full GP (`FTC`) + Gaussian noise + RBF, then Hamiltonian Monte Carlo over `rbf.lengthscale`, `rbf.variance` and `likelihood.variance` ([CIP-003B](../../cip/cip003B.md), [REQ-001C](../../requirements/req001C_posterior-samples-of-gp-hyperparameters.md))

A point estimate hides how well the data pin the hyperparameters down. Here we learn the sinc model, then draw samples of its hyperparameters and read off the uncertainty.

## Build

```sh
make gp
```

## CLI tutorial

```sh
./gp -s 1 learn -# 200 examples/sinc.csv sinc.model
./gp -s 4 sample -n 2000 -b 300 \
    -p rbf.lengthscale=gamma:5,2 \
    -p rbf.variance=gamma:2,10 \
    -p likelihood.variance=gamma:2,500 \
    examples/sinc.csv sinc.model sinc_samples.csv
```

`-n` is the number of samples, `-b` the burn-in during which the step size adapts (those draws are discarded), `-s` the seed. The run prints the acceptance rate (about 0.9 here; the adaptation targets 0.8) and writes `sinc_samples.csv`:

```
iteration,log_posterior,rbf.lengthscale,rbf.variance,likelihood.variance
```

The file is plain CSV, so `pandas.read_csv` or `numpy.loadtxt(..., delimiter=",", skiprows=1)` reads it directly.

### What to expect

With the seed above, the median and 90% interval (5th to 95th percentile) over 2000 draws were:

| Quantity | Median | 90% interval | Learned (MAP) value |
|---|---|---|---|
| `rbf.lengthscale` | 2.30 | 1.86 to 2.76 | 2.24 |
| `rbf.variance` | 0.094 | 0.052 to 0.19 | 0.075 |
| `likelihood.variance` | 0.0023 | 0.0014 to 0.0039 | 0.0020 |

Other seeds change these in the second digit. The signal variance is the least well determined: its interval spans a factor of almost four.

### Things to know

- **Priors are required.** Every sampled parameter needs a prior. Without one the posterior can be improper, and `gp sample` refuses with a message naming the parameters. `--allow-improper` overrides the check.
- **Lengthscale units.** Priors on `rbf.lengthscale` are stated on the lengthscale, as in GPy, and samples are reported as lengthscales. The kernel stores the inverse width \(\gamma = 1/\ell^2\); the prior is converted with the change of variables, so the posterior is the same whichever way you state it. A prior on `rbf.inv_lengthscale` attaches to the stored value instead, and the samples are still reported as lengthscales. With `--unconstrained` the column is the log of the stored width and is labelled `rbf.inv_lengthscale`. The exponential kernel uses \(\ell = 1/\gamma\).
- **Jacobian.** By default the target includes the change-of-variables term for the positive transforms, so the samples follow the priors you state. `--no-jacobian` reproduces GPy, which samples the transformed space without it.
- **Fix parameters** for a run with `--fix rbf.variance`. `--unconstrained` writes the optimiser's unconstrained values instead of natural ones.
- Sparse models are sampled with their inducing points held fixed.

## Python

```python
import numpy as np
import gpc

data = np.loadtxt("examples/sinc.csv", delimiter=",", comments="#")
X, Y = data[:, :-1], data[:, -1:]
m = gpc.models.GPRegression(X, Y, kernel=gpc.kern.RBF(1))
m.optimize(iters=200)
gpc.inference.set_prior(m, "rbf.lengthscale", "gamma", 5.0, 2.0)
gpc.inference.set_prior(m, "rbf.variance", "gamma", 2.0, 10.0)
gpc.inference.set_prior(m, "likelihood.variance", "gamma", 2.0, 500.0)
res = gpc.inference.hmc(m, num_samples=2000, burn_in=300, seed=4)
print(res.accept_rate, res.mean())
res.to_csv("sinc_samples.csv")
```

`hmc` restores the model's parameters afterwards, so the optimised fit is unchanged.

## How we check it

`testHmc` compares the sampler with an analytic Gaussian and with grid integration of a one-parameter GP posterior, checks that the leapfrog energy error scales as \(\epsilon^2\), checks the lengthscale prior integrates to one over the stored width and matches finite differences, and checks the objective gradient (with the Jacobian) against finite differences for exact, sparse and GPLVM targets. `python/tests/test_hmc.py` repeats the grid comparison through the Python façade.
