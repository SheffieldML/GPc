---
id: "0017"
title: "Public gpc surface is discoverable via Python help and docstrings"
status: "Validated"
priority: "Medium"
created: "2026-11-20"
last_updated: "2026-11-20"
related_tenets: ["ecosystem-interoperability", "compose-through-interfaces"]
stakeholders: ["GPy users", "modellers", "teachers", "maintainers"]
tags: ["gpy", "python", "facade", "documentation", "docstrings", "help"]
---

# REQ-0017: Public `gpc` surface is discoverable via Python help and docstrings

## Description

The installable `gpc` Python façade exposes **Python-compatible API documentation**: public classes and methods carry docstrings (or equivalent `__doc__`) so a modeller can use normal Python tooling — `help()`, notebook `?`, IDE hover, and optional autodoc — without opening markdown alone.

Markdown side-by-side porting guides from [REQ-000E](./req000E_gpy-facing-python-facade.md) / [CIP-001D](../cip/cip001D.md) remain the GPy narrative; this requirement owns **interactive / machine-readable** help on the live objects.

**Why this matters**: Ecosystem Interoperability means meeting Python users where they already look for API meaning. Compose Through Interfaces keeps docs on the façade boundary, not inside the SCG loop.

**Who benefits**: GPy users exploring `gpc` in a REPL or IDE; teachers demoing the façade; maintainers checking that bindings stay documented as the surface grows.

## Acceptance Criteria

- [x] Public constructors and methods on the documented first subset (`GPRegression`, `SparseGPRegression`, `gpc.kern` factories, path helpers) have non-empty docstrings visible via `help()` / `__doc__`
- [x] Docstrings state GPy-facing intent briefly (what the call does, key args, units/paths where relevant) without claiming API identity with GPy
- [x] A smoke test or checklist asserts `__doc__` is present for the public subset (fails closed if stripped)
- [x] Optional Sphinx/autodoc generation may consume the same docstrings later; it is not required to close this requirement if `help()` coverage is met
- [x] Non-goals are explicit: full Sphinx site as a close blocker; paramz-style dynamic help trees; putting documentation generation in the optimiser hot path; replacing CIP-001D markdown porting guides

## Notes (Optional)

HOW is [CIP-002F](../cip/cip002F.md) (**Closed**). Complements [REQ-000E](./req000E_gpy-facing-python-facade.md) without reopening CIP-001D.

## References

- **Related Tenets**: ecosystem-interoperability, compose-through-interfaces
- **External Links**: Python `help()`, IPython `?`, Sphinx autodoc / napoleon
- **Related**: [REQ-000E](./req000E_gpy-facing-python-facade.md), [CIP-001D](../cip/cip001D.md)

## Progress Updates

### 2026-11-20
Requirement created after CIP-001D close: interactive help deliberately left out of 001D so documentation stays a separate CIP with a Python-compatible source of truth.

### 2026-11-20 (delivery)
CIP-002F Accepted → Implemented → Closed on `cip002F-python-help-docstrings`. Pybind + pure-Python docstrings; `test_help_docstrings.py` green. Requirement → Validated.
