---
id: "2026-11-26_deepgp-learned-window-width-no-effect"
title: "DeepGP learn_window_width=True leaves the trained ELBO unchanged"
status: "Proposed"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "bugs"
related_cips: ["003C"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- deepgp
- weightspace
---

# Task: DeepGP learned window width has no visible effect

## Description

`gpc.models.DeepGP(..., inducing="weightspace", learn_window_width=True)` adds optimisation parameters (794 against 792 in a two-layer, 16-window model) but the trained ELBO is identical to the fixed-width model to about 1e-10. In the paper-experiment notebooks (`deepgp-mlp/notebooks/05_hidden_layer_inducing`) the "learned windows" row repeats the fixed-window row exactly (composite: RMSE 0.197 and ELBO -400.669 in both; three-level: 0.325 and -645.841 in both).

Reproduction (gpc built from `main` at 48661ce):

```python
X = np.random.randn(300, 4); Y = np.sin(X[:, :1])
for flag in (False, True):
    m = DeepGP(X, Y, layers=2, hidden=3, num_inducing=16, kernel="mlp",
               inducing="weightspace", learn_window_width=flag)
    m.optimize(iters=300, batch_size=100, samples=1, adam_lr=1e-2, seed=0)
    print(flag, m.elbo())   # 58.26947220448 (True) vs 58.26947220449 (False)
```

Possibilities: the width gradient is zero or not wired into the deep GP objective; the width sits at a bound of its sigmoid-packed range; or the Adam update does not reach it. `SparseGPRegression(..., learn_window_width=True)` should be checked as a control.

## Acceptance Criteria

- [ ] Cause identified
- [ ] A test shows the learned width moves from its initial value on a problem where it should
- [ ] `checkGradients`-style check of the width gradient in the deep GP objective

## Related

- CIP: 003C (learned windows on `DeepGP`)

## Progress Updates

### 2026-11-26

Task created with Proposed status.
