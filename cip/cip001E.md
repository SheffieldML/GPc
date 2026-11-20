---
author: "Neil Lawrence"
created: "2026-11-18"
id: "001E"
last_updated: "2026-11-18"
status: "Implemented"
compressed: false
related_requirements: ["000F", "0008", "000E"]
related_cips: ["0014", "001D", "0010"]
tags:
- cip
- optimiser
- display
- observer
- paramz
- api
- python
title: "Interceptable optimiser events and structured model display"
---

# CIP-001E: Interceptable optimiser events and structured model display

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md). It reuses path/unit policy from [CIP-0014](./cip0014.md) / [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md) for description rows, and supplies the C++ hooks that [CIP-001D](./cip001D.md) / [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) bind for live progress and rich repr. It does **not** change `.model` stream formats ([REQ-000A](../requirements/req000A_shared-model-interchange.md) / CIP-0016 stay orthogonal).

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented (worktree `cip001E-interceptable-display`)
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Backlog

- [Phase A — optimiser observer](../backlog/features/2026-11-18_cip001E-phase-a-optimiser-observer.md) (High, Completed)
- [Phase B — model description](../backlog/features/2026-11-18_cip001E-phase-b-model-description.md) (High, Completed)
- [Phase C — Python progress + repr](../backlog/features/2026-11-18_cip001E-phase-c-python-progress-repr.md) (High, Completed; also CIP-001D B2)

## Summary

Replace hard-coded optimiser `cout` and ad-hoc `display` printing with a **data-first C++ protocol**: typed `OptimiserEvent` + observer on `COptimisable`, and a structured `ModelDescription` + text formatter for GP / kernel / noise. Default observers/formatters preserve CLI behaviour. The Python façade consumes the same surfaces thinly (`messages` / `callback`, `__str__` / `__repr__`) without cloning paramz’s `Parameterized` tree.

**Which requirements?** Primary: [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md). Reuses [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md). Enables live progress and repr for [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md).

## Motivation

Optimiser loops in [`COptimisable.cpp`](../COptimisable.cpp) print via `cout` gated by an integer verbosity; there is no callback, stream injection, or typed progress record. Model/`CKern`/`CNoise` `display` methods hand-write text; [`CGp::display`](../CGp.cpp) mixes `cout` with the passed `ostream`, so redirection is broken. Display ignores GPy-canonical paths/units already available via `paramPaths()`.

GPy/`paramz` users expect interceptable progress and printable models. We want that ergonomics idea, not a second parameter object graph. Without a C++ protocol, CIP-001D cannot offer notebook progress or stable repr without inventing a parallel logging path.

## Detailed Description

### Design principles

1. **Data first, renderers second.** Loops emit events; models build descriptions; text/HTML/Python are consumers.
2. **Indexed optimisation stays the hot path.** Observers fire only at outer-iteration / terminal boundaries — same cadence as today’s `verbosity>2` prints — never from `computeObjectiveVal` / `computeObjectiveGradParams` / kernel `compute` / BLAS.
3. **Compose through interfaces.** New contracts live on `COptimisable` and displayable model types; drivers do not special-case Python.
4. **GPy path/units for description rows.** Description uses CIP-0014 canonical paths and natural units; no second naming table.
5. **Persistence ≠ display.** `writeParamsToStream` / `.model` unchanged.

### Phase A — OptimiserEvent and observer

Introduce a small POD (names may tighten at implement time):

```cpp
enum class OptimiserPhase { Start, Iter, Converged, MaxIters, Failed };

struct OptimiserEvent {
  OptimiserPhase phase;
  int optimiser;       // existing enum / id
  int iter;
  int funcEval;        // -1 if not tracked by this driver
  double objective;
  double gradNorm;     // NaN if not computed for this event
  double stepOrScale;  // e.g. SCG lambda; NaN if N/A
  std::string message; // warnings / convergence notes; may be empty
};
```

On [`COptimisable`](../COptimisable.h):

- Hold zero or more observers (`std::function<void(const OptimiserEvent&)>` and/or a small `IOptimiserObserver` virtual interface — prefer `std::function` unless multiple inheritance force a vtable).
- Helper `notify(const OptimiserEvent&)` called from SCG, CG, GD, LBFGS wrappers at outer-iteration and terminal points.
- **Default observer:** `TextProgressObserver` writes to a configurable `ostream*` (default `&std::cout`), gated by existing `verbosity` so CLI behaviour matches today’s useful lines.
- Unconditional warning `cout`s move onto the event channel (`phase=Failed` / `MaxIters` with `message`) so quiet runs stay quiet when verbosity is low / observers off.

**Rejected for v1:** firing from every line-search probe; allocating format strings in the objective; requiring Python to be linked into the C++ core.

### Phase B — ModelDescription and formatters

Introduce a structured description (names TBD):

```cpp
struct ParamRow {
  std::string path;      // GPy-canonical
  double value;          // natural units
  std::string transform; // e.g. "exp", "none"; empty OK
  bool fixed;
  std::string role;      // optional: "kern", "noise", "inducing", …
};

struct ModelDescription {
  std::string title;           // e.g. "Standard GP Model" / sparse approx name
  std::vector<std::string> headerLines; // approx, n, optimiser, …
  std::vector<ParamRow> params;
  // optional: logLikelihood present flag + value
};
```

- Builders: `CKern::describe()`, `CNoise::describe()` (or free functions), `CGp::describe()` that walks kern/noise via existing path APIs and fills header fields.
- `TextFormatter::format(const ModelDescription&, ostream&)` produces the multi-line summary.
- `display(ostream&)` becomes `format(describe(), os)` for GP / kern / noise used by CLI.
- **Fix:** [`CGp::display`](../CGp.cpp) must write only to `os` (no mixed `cout`).

**Rejected for v1:** full paramz tree; HTML formatter required in C++ (optional later for `_repr_html_`); changing stream key names.

### Phase C — CIP-001D consumption (thin Python)

Documented here and mirrored in CIP-001D amendments:

- pybind11 exposes `OptimiserEvent` as a read-only object.
- `optimize(..., messages=True)` / `optimize(..., callback=fn)` installs a C++ observer holding `py::function`, invoked **only** on outer-iteration / terminal events.
- Soften CIP-001D’s absolute “no Python callbacks from SCG” to: *no callbacks inside objective/gradient evaluation, kernel `compute`, or sparse factorisation; optional outer-iteration observer allowed*.
- GIL: acquire around the Python call only if optimise released the GIL; document that a slow callback stalls the next iteration (SciPy/`paramz` contract).
- Opt-out free: `messages=False` and no `callback` → zero Python traffic during the run.
- `__str__` / `__repr__` call `describe()` (or bound text format once); optional `_repr_html_` after a simple HTML formatter or Python-side table — not a C++ Phase A/B close blocker.

Illustrative surface:

```python
m.optimize(max_iters=100, messages=True)
m.optimize(callback=lambda e: print(e.iter, e.objective))
print(m)
```

### Tenet trade-offs

- **Flexibility vs Compose:** observers add a virtual/call boundary only at iteration edges, not element-wise.
- **Ecosystem vs clone:** match paramz interceptability and printable models; reject Parameterized graph.
- **Assert:** tests must capture events/descriptions without relying on scraping `cout`.

## Implementation Plan

1. **Phase A — Optimiser observer**
   - Add `OptimiserEvent` + observer registration on `COptimisable`.
   - Migrate SCG / CG / GD / LBFGS progress and budget warnings to `notify`.
   - Default `TextProgressObserver` + verbosity gating; document in [docs/optimisers.md](../docs/optimisers.md).
   - Extend [`testOptimisable`](../testOptimisable.cpp) (CIP-0010) to collect events on a tiny problem.

2. **Phase B — Model description**
   - Add `ModelDescription` / `ParamRow` and builders for kern, noise, `CGp`.
   - Text formatter; migrate `display`; fix `CGp::display` stream bug.
   - Unit/golden tests for description paths/values and formatter smoke.

3. **Phase C — Python bindings (with CIP-001D)**
   - Bind event POD; wire `messages` / `callback` on façade `optimize`.
   - `__str__` / `__repr__` from description; optional HTML later.
   - Amend CIP-001D design decision #4 and API sketch (done in same design pass as this Proposed CIP).

## Backward Compatibility

- CLI verbosity defaults and typical progress *content* stay familiar; exact string layout may change slightly when routed through the default observer — document any intentional differences.
- `display(ostream&)` signatures remain; behaviour becomes stream-correct for `CGp`.
- No `.model` / stream field changes.
- Existing `setVerbosity` continues to control the default text observer.

## Testing Strategy

- **Optimiser:** register a collecting observer; assert `Start` / `Iter` / terminal phases, monotone `iter`, finite `objective` on a tiny `COptimisable` (reuse CIP-0010 harness).
- **Silent:** verbosity 0 / no extra observers → no progress lines on a redirected stream.
- **Description:** known RBF (+ compound) and GP fixture: `paramPaths()` match description paths; values match path get in natural units.
- **Display:** `CGp::display(oss)` contains kernel/noise lines and does not require reading `cout`.
- **Python (CIP-001D):** callback invoked at least once per outer iteration when messages/callback set; with both off, callback not called.

## Related Requirements

- [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md) — primary WHAT
- [REQ-0008](../requirements/req0008_gpy-like-parameter-access.md) — path/unit policy for description rows
- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) — live progress and repr consumers

## Implementation Status

- [x] Phase A: OptimiserEvent + observer; migrate optimiser prints; testOptimisable event collection; docs/optimisers.md note
- [x] Phase B: ModelDescription + text formatter; migrate display; fix CGp stream bug; description/display tests
- [x] Phase C: CIP-001D binds messages/callback and __str__/__repr__ (tracked also on CIP-001D)
- [ ] Close: REQ-000F acceptance criteria checked; compress into docs (optimisers + display/param-paths pointer); mark Implemented → Closed after review

## References

- [REQ-000F](../requirements/req000F_interceptable-optimiser-model-display.md)
- [CIP-0014](./cip0014.md) / [docs/param-paths.md](../docs/param-paths.md)
- [CIP-001D](./cip001D.md) (Python consumer)
- [CIP-0010](./cip0010.md) (testOptimisable)
- [docs/optimisers.md](../docs/optimisers.md)
- GPy/`paramz` messages and SciPy `minimize(callback=...)`
