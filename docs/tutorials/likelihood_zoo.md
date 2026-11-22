# Tutorial: GPy-aligned likelihood zoo (CIP-003A)

Observation families beyond Gaussian and Bernoulli–probit live as `CNoise`
subclasses with Laplace / Gauss–Hermite oracles and a thin `gpc.likelihoods`
façade ([REQ-001B](../../requirements/req001B_gpy-aligned-likelihood-zoo.md)).

## C++ density / VE

```cpp
CStudentTNoise noise(&Y);
noise.setVariance(0.25);
noise.setDegFree(6.0);
double gMu = 0.0, gVar = 0.0;
const double ell = noise.expectedLogLikelihood(y, mu, var, gMu, gVar);
```

Pointwise Laplace pieces for mode finding go through
`ndlnoise::laplacePointDerivs(noise, y, f, logp, d1, W, dW)`.

## Dense Laplace on `CGp`

```cpp
CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
model.setLaplaceInference(true);
const double L = model.logLikelihood();
```

## Python

```python
import gpc
import numpy as np

X = np.linspace(-1, 1, 20)[:, None]
Y = 0.3 * X + 0.05 * np.random.randn(20, 1)
lik = gpc.likelihoods.StudentT(variance=0.25, deg_free=6.0)
m = gpc.models.GPRegression(X, Y, kernel="rbf", likelihood=lik, inference="laplace")
m.optimize(iters=50)
mu, var = m.predict_noiseless(X)
```

DeepGP accepts the same type strings (`student_t`, `poisson`, …) when the
layer likelihood uses the GH `expectedLogLikelihood` path.

## Related

- [CIP-003A](../../cip/cip003A.md), [svgp](./svgp.md), [cgp_ep_classification](./cgp_ep_classification.md)
- `./testLikelihoodZoo`, `python/tests/test_likelihoods.py`
