---
author: "Neil Lawrence"
created: "2026-11-20"
id: "002F"
last_updated: "2026-11-20"
status: "Closed"
compressed: true
related_requirements: ["0017", "000E"]
related_cips: ["001D"]
tags:
- cip
- python
- facade
- documentation
- docstrings
- help
title: "Python-compatible help and docstrings for the gpc façade"
---

# CIP-002F: Python-compatible help and docstrings for the `gpc` façade

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md). It extends the façade closed in [CIP-001D](./cip001D.md) / [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) without reopening that CIP’s modelling scope.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [x] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Backlog

- [Python help/docstrings](../backlog/features/2026-11-20_cip002F-python-help-docstrings.md) (High, Completed)

## Summary

Make the public `gpc` surface discoverable through **normal Python tooling**: non-empty `__doc__` / docstrings on public classes and methods so `help(gpc.GPRegression)`, notebook `?`, and IDE hover work. Keep CIP-001D markdown side-by-side docs as the GPy porting narrative; do not treat Sphinx publication as a close blocker.

**Which requirements?** Primary: [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md). Complements [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md).

## Motivation

CIP-001D closed with markdown porting guides and a GPy-shaped API (`gpc.kern`, `gpc.models`). Constructor-level pybind strings exist in places, but method-level and class-level help is thin or missing. GPy users expect to poke objects in a REPL. A separate documentation CIP keeps “porting narrative” (markdown) distinct from “Python-compatible API meaning” (`__doc__`).

## Detailed Description

### Design decisions (for Proposed review)

1. **Source of truth is `__doc__`.** Acceptance requires that Python can load documentation from the live objects. Markdown alone is insufficient for this CIP.
2. **Where to put strings.** Prefer docstrings on pure-Python wrappers (`gpc.models`, `gpc.kern`) and pybind `.def(..., "doc")` / class docs for `_gpc` types that remain public. Either path is fine if `help()` works after `import gpc`.
3. **First subset only.** Cover the documented REQ-000E / CIP-001D subset: `GPRegression`, `SparseGPRegression`, kern factories, path helpers, `optimize` / `predict` / `objective`. Expand when new façade surfaces land.
4. **No paramz clone.** Docstrings explain calls; they do not implement GPy’s Parameterized help tree.
5. **Sphinx optional.** Autodoc may consume the same strings later (compression); not a close criterion.
6. **Orthogonal to CIP-001E.** Structured `__repr__` / optimiser messages remain CIP-001E / Phase B2 deferred work.

### Rejected / deferred

| Idea | Status |
|---|---|
| Folding help into CIP-001D close | Rejected — 001D Closed; this CIP owns help |
| Full Sphinx site as close blocker | Deferred / optional compression |
| Regenerating docs inside SCG | Rejected |

## Implementation Plan

1. Inventory public `gpc` names vs empty/`None` `__doc__`.
2. Add class/method docstrings (pure Python and/or pybind).
3. Pytest: assert `__doc__` non-empty for the public subset.
4. Point `python/README.md` at `help()` as the interactive path; keep side-by-side markdown for porting.

## Backward Compatibility

- No modelling behaviour change; documentation only.
- Existing scripts unaffected.

## Testing Strategy

- Unit: `__doc__` presence smoke for public API.
- Manual: `help(gpc.GPRegression)`, `help(gpc.kern.RBF)` in a venv after install/build.

## Related Requirements

- [REQ-0017](../requirements/req0017_python-facade-help-docstrings.md) — primary WHAT
- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) — façade subset this documents

## Implementation Status

- [x] Inventory + docstring pass (pybind model methods + `gpc.kern` / `gpc.models`)
- [x] `__doc__` smoke tests (`python/tests/test_help_docstrings.py`)
- [x] README pointer
- [x] Close after local pytest green (`compressed: false` until formal doc compression)

## References

- [CIP-001D](./cip001D.md) (Closed)
- [python/docs/gpy_side_by_side.md](../python/docs/gpy_side_by_side.md)
- Python `help()`, IPython `?`, Sphinx autodoc
