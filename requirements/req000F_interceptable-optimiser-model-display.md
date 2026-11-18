---
id: "000F"
title: "Callers can intercept optimiser progress and render structured model descriptions"
status: "Implemented"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets: ["compose-through-interfaces", "ecosystem-interoperability", "flexibility-with-performance", "assert-and-verify"]
stakeholders: ["maintainers", "modellers", "GPy users", "teachers"]
tags: ["optimiser", "display", "observer", "paramz", "api", "python", "notebook"]
---

# REQ-000F: Callers can intercept optimiser progress and render structured model descriptions

## Description

GPc exposes **structured, interceptable surfaces** for optimiser progress and model display so CLI text, tests, batch runs, and language façades share one C++ source of truth. Optimiser loops emit typed progress events that observers can consume; models expose a structured description (paths, values, constraints, header metadata) that formatters render. Default text behaviour remains available for today’s CLI and demos.

The Python façade ([REQ-000E](./req000E_gpy-facing-python-facade.md)) may attach a **live outer-iteration** progress callback and render the same description via `__str__` / `__repr__` (HTML optional). Parameter paths and units stay aligned with [REQ-0008](./req0008_gpy-like-parameter-access.md). Persistence (`.model` / `CStreamInterface`) stays separate from display.

**Why this matters**: Today optimiser progress and `display` are presentation-coupled (`cout` and hand-written `ostream` text), so quiet batch runs, golden tests, notebook progress bars, and a thin Python wrapper cannot intercept cleanly. Compose Through Interfaces wants contracts at model boundaries; Flexibility Without a Performance Tax forbids putting formatting or interpreters in likelihood/kernel/BLAS work; Interoperate With the GP Ecosystem means matching the useful part of paramz/GPy ergonomics (named, interceptable) without cloning the `Parameterized` tree; Assert and Verify needs capturable, testable output.

**Who benefits**: CLI and demo users (unchanged text); maintainers writing tests; notebook users and the Python façade; anyone wrapping GPc from another language.

## Acceptance Criteria

- [ ] Optimiser drivers expose typed progress events at outer-iteration and terminal boundaries (not inside objective/gradient evaluation, kernel `compute`, or BLAS)
- [ ] Callers can register an observer / callback for those events; with no observer (or verbosity/messages off), runs produce no progress traffic
- [ ] Default text progress reproduces today’s useful CLI lines (iteration, objective, and driver-specific scale/step where already shown) via a default observer, not hard-coded `cout` scattered as the only path
- [ ] Models (at least GP, kernel, noise used by the CLI) expose a structured description built from GPy-canonical paths/units consistent with [REQ-0008](./req0008_gpy-like-parameter-access.md), plus documented header fields (model class, approx, sizes, optimiser name, log-likelihood when cheap/safe)
- [ ] Text display (`display(ostream&)`, CLI `display`) is a formatter over that description and writes only to the provided stream (no mixed `cout` leakage)
- [ ] `.model` / stream I/O field names and semantics are unchanged
- [ ] Automated checks cover: event fields on a tiny optimisable; structured description paths/values for a known kernel/GP; text formatter smoke (or golden) against a fixture; silent run with observers off
- [ ] The Python façade ([REQ-000E](./req000E_gpy-facing-python-facade.md)) can: attach a live outer-iteration callback (and/or `messages=` text), and render the description via `__str__` / `__repr__`; optional `_repr_html_` is allowed but not required for first close of this requirement’s C++ core
- [ ] Non-goals are explicit: full GPy/`paramz` `Parameterized` tree; regex parameter indexing; tied-parameter UI; changing optimiser budget semantics ([docs/optimisers.md](../docs/optimisers.md)); putting Python or HTML assembly inside objective/gradient evaluation

## Notes (Optional)

HOW is [CIP-001E](../cip/cip001E.md) (Implemented). Extends [REQ-0008](./req0008_gpy-like-parameter-access.md) for display paths/units and feeds [REQ-000E](./req000E_gpy-facing-python-facade.md) / [CIP-001D](../cip/cip001D.md) for live progress and rich repr. Orthogonal to [REQ-000A](./req000A_shared-model-interchange.md) (file interchange ≠ live display).

## References

- **Related Tenets**: compose-through-interfaces, ecosystem-interoperability, flexibility-with-performance, assert-and-verify
- **External Links**: GPy/`paramz` optimiser messages and model printing; SciPy `minimize` `callback=`

## Progress Updates

### 2026-11-18
Requirement created as Proposed WHAT after design discussion: data-first optimiser events and model description; thin Python live progress in scope as outer-iteration observer consumer.

### 2026-11-18 (accept)
CIP-001E Accepted; requirement moved to Ready pending backlog + implementation on worktree `cip001E-interceptable-display`.

### 2026-11-18 (implemented)
CIP-001E Phases A–C landed on `cip001E-interceptable-display`: C++ observer + ModelDescription, Python `messages`/`callback` and `__str__`. Requirement → Implemented; await Close after review.
