---
author: "Neil Lawrence"
created: "2026-11-20"
id: "002E"
last_updated: "2026-11-20"
status: "Closed"
compressed: false
related_requirements: ["0002", "0001", "000E"]
related_cips: ["0019", "001A", "002C", "002D", "002B"]
tags:
- cip
- ci
- github-actions
- infrastructure
- wheels
- numerical-stability
title: "Tier CI triggers — hard gates every PR; numerics/wheels path or special"
---

# CIP-002E: Tier CI triggers — hard gates every PR; numerics/wheels path or special

> CIPs describe HOW to achieve requirements (WHAT). This CIP only changes **when** existing workflows run. It does not change test content, soft→hard soak criteria for [CIP-002C](./cip002C.md) wheels, or [CIP-0021](./cip0021.md) CMake cutover.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [x] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Stop running every Actions workflow on every push/PR. Keep **Compile** and **Reference tests** as the default hard merge gates; keep **Numerical stability** and **Bench** on path filters (already done); demote **Coverage** and **Wheels** to path / main / schedule / `workflow_dispatch` so PR CI minutes track what the change can break.

**Which requirements?** [REQ-0002](../requirements/req0002_runnable-verification.md) (trustworthy verification without redundant always-on cost); [REQ-0001](../requirements/req0001_contemporary-build.md) (build still gated); [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) packaging delivery checked when the packaging surface changes ([CIP-002C](./cip002C.md)).

## Motivation

Seven workflows fire (or can fire) around `main`. Numerical stability and bench already use path filters. Compile, Reference, Coverage, and Wheels still run on every PR — including CIP/docs-only changes. Coverage largely re-runs matio under gcov. Wheels (cibuildwheel Linux + macOS) prove *delivery*, not model correctness; [CIP-002C](./cip002C.md) already says makefile Compile / Reference / Coverage remain the hard native gates, and **Publish wheels** is already dispatch/release-only.

Assert and Verify still holds: the hard gate stays Reference (+ Compile). Flexibility Without a Performance Tax applies to CI too — do not pay cibuildwheel on a `CKern` tweak or a backlog markdown edit.

## Detailed Description

### Trigger tiers

| Tier | Workflows | When |
| --- | --- | --- |
| **1 — Every PR / push to `main` (hard)** | Compile; Reference tests | Always, with `paths-ignore` for pure governance (`docs/**`, `cip/**`, `backlog/**`, `tenets/**`, `coe/**`, `requirements/**`, `**/*.md`) |
| **2 — Path-triggered** | Numerical stability; Bench timing; Wheels | Existing NS/bench globs; wheels: `python/**`, root `CMakeLists.txt`, `wheels.yml` (+ `workflow_dispatch`) |
| **3 — Main / schedule / dispatch** | Coverage | `push` to `main` + `workflow_dispatch` (no every-PR) |
| **4 — Release / dispatch only** | Publish wheels | Unchanged (`workflow_dispatch` + `release`) |

Windows MSVC soft job remains in Compile (runs when tier-1 fires). Job-level Windows path filter deferred.

### Numerics vs others

Path filters are the signal. Numerical stability already keys off LAPACK-bound / GP core sources (`CMatrix`, `CKern`, `CNoise`, `CGp`, adjoints, `ndlutil`/`ndlfortran`, `numerical_stability/**`, make fragments). That is sufficient: false positives on core `.cpp` are cheap; false negatives on docs/CIP-only PRs are intended. Keep `workflow_dispatch` for pre-release full soaks when paths did not match.

### Wheels as a special trigger

**Demoted now** (Accept decision): `wheels.yml` is path + dispatch; soft gate (`continue-on-error`) unchanged until CIP-002C soak closes. Do not require wheel builds to merge C++-only PRs. Publish stays tier 4.

### Non-goals

- Merging workflows into one mega-YAML
- Changing soft→hard for Windows ([CIP-002B](./cip002B.md)) or wheels beyond trigger policy
- Replacing matio Reference with numerical stability (orthogonal; [CIP-0019](./cip0019.md))
- CMake CI cutover ([CIP-0021](./cip0021.md))
- Job-level path filter for Windows soft (deferred)

## Implementation Plan

1. Document the four tiers in a short comment block at the top of each affected workflow.
2. Add `paths-ignore` (governance + markdown) to Compile and Reference.
3. Change Coverage `on:` to `push: main` + `workflow_dispatch`.
4. Demote `wheels.yml` now: packaging path filter + `workflow_dispatch` (soft gate kept).
5. Windows soft: leave on Compile; job-level path filter deferred.
6. Spot-check on next PRs / dispatches (see Testing Strategy).

## Backward Compatibility

No runtime or API change. Contributors may see fewer checks on a PR; merge policy remains “Compile + Reference green” (when those workflows run). Maintainers can still force Coverage / NS / Wheels via Actions UI. Force a wheel build anytime with `workflow_dispatch`.

## Testing Strategy

- Docs/CIP-only PR: Compile and Reference skipped via `paths-ignore`.
- `CGp.cpp` PR: Compile + Reference (+ NS/bench if paths match); no Wheels; no Coverage.
- `python/` or root `CMakeLists.txt` PR: Wheels soft gate runs.
- Push to `main`: Coverage runs; Wheels only if packaging paths changed.
- Publish unchanged on `workflow_dispatch` / release.

## Related Requirements

- [REQ-0002](../requirements/req0002_runnable-verification.md)
- [REQ-0001](../requirements/req0001_contemporary-build.md)
- [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md)

## Implementation Status

- [x] Tier map recorded on workflows
- [x] Coverage demoted off every-PR
- [x] Wheels demoted now (soft gate remains; packaging paths only)
- [x] Compile/Reference `paths-ignore` for governance/markdown
- [ ] Windows job-level path filter (deferred)
- [x] Spot-check PRs / dispatches as in Testing Strategy (close after green on Actions)

## References

- Workflows: `.github/workflows/{compile,reference-tests,coverage,numerical-stability,bench-timing,wheels,publish-wheels}.yml`
- [CIP-0019](./cip0019.md) / [CIP-001A](./cip001A.md) — NS already path-filtered
- [CIP-002C](./cip002C.md) — wheels soft gate; Publish already special
- [CIP-002D](./cip002D.md) — bench timing path-filtered
- Session discussion 2026-11-20: split CI occasions; wheels as special trigger
