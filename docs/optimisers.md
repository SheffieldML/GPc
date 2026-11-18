# Optimisers and `-#` / `max_iters` budgets

How GPc’s CLI `-#` / `setMaxIters` maps onto each driver, and how that relates to GPy/`paramz`.

## GPy / paramz (what “GPy standard” means)

`m.optimize(..., max_iters=N)` in GPy goes through `paramz.optimization.Optimizer`. The **docstring** often says `max_iters` is a maximum number of **function evaluations**. The **code** is optimiser-specific:

| Optimiser | paramz / SciPy behaviour |
|---|---|
| **L-BFGS-B** (`lbfgs` / `bfgs` in many examples) | `fmin_l_bfgs_b(..., maxfun=N, maxiter=N)` — **both** caps set to `N` |
| **SCG** | Outer loop `while iteration < maxiters` — **algorithm iterations**. `max_f_eval` is accepted by the SCG helper but is **not** what stops the loop in current paramz SCG |
| **TNC / simplex / …** | Their own SciPy `maxfun` / `maxiter` mappings |

So “converge to GPy” for L-BFGS means honouring a shared budget on **iterations and function evaluations**. For SCG, matching GPy means keeping an **iteration** budget (despite the misleading docstring).

GPy L-BFGS is still Fortran under SciPy; the **Python reverse-communication loop** enforces `maxiter` / `maxfun`. GPc uses Nocedal `lbfgs_` in `ndlfortran.f` the same way — the budget must live in the **C++** wrapper, not in the Fortran subroutine (which has no max-iter argument).

## GPc mapping (`-#` → `setMaxIters`)

| Driver (`-O`) | What `-# N` / `setMaxIters(N)` limits | Notes |
|---|---|---|
| `scg` (default) | **Algorithm iterations** | Matches paramz SCG. Each iteration may evaluate the objective/gradient more than once; we do **not** reinterpret `-#` as function evaluations (that would stop earlier than GPy SCG for the same `N`). |
| `conjgrad` | Algorithm iterations (and optional f-eval terminate flags) | Rasmussen-style CG |
| `graddesc` | Algorithm iterations | |
| `quasinew` / `lbfgs` | **Both** reverse-communication steps **and** function evaluations, capped at `N` | Matches GPy `opt_lbfgsb` (`maxiter=N`, `maxfun=N`) |

Help text historically said `-#` was “for scaled conjugate gradient”; it now applies to all drivers as in the table above.

## Why SCG stays iteration-based

Making SCG treat `-#` as function evaluations is easy mechanically (`funcEval` + break), but it would **diverge** from paramz SCG for the same numeric `max_iters`, and would change tutorial recipes (`-# 100` would mean fewer SCG steps). Prefer documenting GPy’s docstring vs SCG reality over changing SCG semantics.

If a hard **function-evaluation** cap is needed for SCG later, expose it explicitly (e.g. enable `setFuncEvalTerminate(true)` / `setMaxFuncEvals`) rather than overloading `-#`.

## Progress observers (CIP-001E)

Optimiser drivers emit typed `OptimiserEvent` values at **outer-iteration** and terminal boundaries via `notifyOptimiserEvent`. Callers can register `addOptimiserObserver(...)`; default verbosity-gated text goes to `setOptimiserProgressStream` (default `cout`) when `setDefaultTextProgress(true)`.

Events are **not** fired from inside `computeObjectiveVal` / `computeObjectiveGradParams`. Verbosity 0 keeps the default text path silent. Python `messages=` / `callback=` (CIP-001D Phase B2) bind the same observer.

## Related

- Implementation: `COptimisable::{scgOptimise,lbfgsOptimise,cgOptimise,gdOptimise}`
- Fortran: `ndlfortran.f` `SUBROUTINE LBFGS` (Nocedal)
- GPy entry: `GPy.core.GP.optimize` → `paramz.optimization.opt_lbfgsb` / `opt_SCG`
- CIP-001E / REQ-000F
