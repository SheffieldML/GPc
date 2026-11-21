# Dense EP classification on `CGp` (CIP-0033 Phase B)

**GPy counterpart:** `GPClassification` = Bernoulli + Probit + EP.  
**Status:** FTC-only on `CGp` via `setEpInference(true)`. IVM remains the sparse / active-set route ([ivm_unitsquare](./ivm_unitsquare.md)).

## Build and smoke test

```sh
make testEp && ./testEp
```

`testEp` checks `ep_moments`, a frozen numpy twin of the EP marginal ([`cip/cip0033/ep_ftc_fixture.txt`](../../cip/cip0033/ep_ftc_fixture.txt)), and a small `CGp` FTC + EP optimise with FD kernel grads.

## C++ sketch

```cpp
CRbfKern kern(X);
CProbitNoise noise(&Y);   // labels in {0,1} or {-1,1}
CGp model(&kern, &noise, &X, CGp::FTC, 0, 0);
model.setEpInference(true);
model.optimise(100);
double L = model.logLikelihood();  // GPy-style EP log Z + kernel prior
```

No auto White: softness lives in the EP sites, not observation noise in \(K\).

## When to use IVM instead

| Path | Use when |
|---|---|
| `CGp` + EP (this page) | Dense \(N\) small–moderate; GPy tutorial parity |
| IVM + probit | Large \(N\); active-set selection; White softener (CIP-0027 exception) |

Laplace on `CGp` is Phase C of [CIP-0033](../../cip/cip0033.md).
