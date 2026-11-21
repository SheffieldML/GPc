# Dense EP / Laplace classification on `CGp` (CIP-0033)

**GPy counterpart:** `GPClassification` = Bernoulli + Probit + EP or Laplace.  
**Status:** FTC-only on `CGp` via `setEpInference(true)` or `setLaplaceInference(true)` (mutually exclusive). IVM remains the sparse / active-set route ([ivm_unitsquare](./ivm_unitsquare.md)).

## Build and smoke test

```sh
make testEp && ./testEp
```

`testEp` checks `ep_moments`, Laplace Hessian oracles, frozen numpy twins ([`ep_ftc_fixture.txt`](../../cip/cip0033/ep_ftc_fixture.txt), [`laplace_ftc_fixture.txt`](../../cip/cip0033/laplace_ftc_fixture.txt)), and small `CGp` FTC optimises with FD kernel grads for both methods.

## C++ sketch

```cpp
CRbfKern kern(X);
CProbitNoise noise(&Y);   // labels in {0,1} or {-1,1}
CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
model.setEpInference(true);       // or: model.setLaplaceInference(true);
model.optimise(100);
double L = model.logLikelihood(); // approx log Z + kernel prior
```

No auto White: softness lives in EP sites / the Laplace Hessian, not observation noise in \(K\).

## When to use which path

| Path | Use when |
|---|---|
| `CGp` + EP | Dense \(N\); GPy default classification parity |
| `CGp` + Laplace | Dense \(N\); cheaper mode-based approx; often close to EP |
| IVM + probit | Large \(N\); active-set selection; White softener (CIP-0027 exception) |
