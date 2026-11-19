# Tutorial: Stochastic Variational GP (SVGP)

CIP-002A adds `CSvgp`: an uncollapsed sparse variational GP with whitened
\(q(\mathbf{u})\), CVI-form natural gradients, and Adam on seeded minibatches.

## Oracle (Phase 1)

With Gaussian noise, the full batch and step size \(\gamma=1\), one natural-gradient
step from the prior lands on the Titsias optimum and the ELBO matches `CGp`
DTCVAR `logLikelihood` to round-off:

```cpp
CSvgp sv(&kern, &X, &y, Z, beta);
sv.naturalGradientStep(1.0);
// sv.elbo() ≈ gp.logLikelihood() for the same Z / kern / beta
```

Run `make testSvgp && ./testSvgp` for the full gate (including singular \(K_{uu}\)).

## Gradients and Adam (Phases 2–3)

Hyperparameter / \(\mathbf{Z}\) / \(\beta\) gradients hold whitened \(q(\mathbf{v})\)
fixed and route through `CCholAdjoint`. Pack unconstrained parameters with
`getOptParams` / `setOptParams`, then:

```cpp
sv.setNatGradGamma(0.5);   // interleaved natgrad after each Adam step
CAdam adam(0.05);
adam.setBatchSize(32);
adam.setSeed(42);
adam.setMaxIters(200);
adam.maximise(sv);
```

Same seed ⇒ identical trajectory. Progress hooks use
`sv.setProgressObserver(...)` (`StochasticOptimiserEvent`).

## Non-Gaussian (Phase 4)

```cpp
CProbitNoise noise(&y);
sv.setLikelihood(&noise);
sv.naturalGradientStep(0.2);  // warm up γ < 1 for non-conjugate
```

`CNoise::expectedLogLikelihood` is \(\mathbb{E}_q[\log p(y\mid f)]\) (Gauss–Hermite
for probit), distinct from ADF `logLikelihood(mu, var, y)`.

## Related

- CIP-002A, REQ-0015
- Collapsed alternative: `CGp` DTCVAR; sparse classification alternative: CIP-0026 PPA
