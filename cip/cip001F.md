---
author: "Neil Lawrence"
created: "2026-11-18"
id: "001F"
last_updated: "2026-11-18"
status: "Proposed"
compressed: false
related_requirements: ["0010"]
related_cips: ["0009", "0013", "001D", "001E"]
tags:
- cip
- concurrency
- threading
- documentation
- rng
- blas
- python
- infrastructure
title: "L2/L3 concurrency contract: docs, RNG hygiene, BLAS nesting"
---

# CIP-001F: L2/L3 concurrency contract: docs, RNG hygiene, BLAS nesting

> CIPs describe HOW to achieve requirements (WHAT). This CIP implements [REQ-0010](../requirements/req0010_documented-concurrency-contract.md). It extends the L1 note from [CIP-0009](./cip0009.md) without reopening that closed CIP. [CIP-001D](./cip001D.md) / [CIP-001E](./cip001E.md) consume the contract for Python docs and GIL policy. L4 intra-eval parallelism ([CIP-0025](./cip0025.md); was CIP-0013 P7) stays orthogonal.

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Publish and enforce a **process-level concurrency contract**: keep one model graph single-threaded; allow independent instances once process globals are named and made safe or documented; state BLAS nesting rules; align the Python façade. Do **not** make a single `CGp` reentrant or mutexed.

**Which requirements?** [REQ-0010](../requirements/req0010_documented-concurrency-contract.md).

## Motivation

[CIP-0009](./cip0009.md) closed with explicit caches and the rule “one model instance remains single-threaded” in [docs/numerics.md](../docs/numerics.md). That is the correct L1 baseline for a library that stores factorisations in `mutable` members for `const` `COptimisable` entry points.

Climbing further toward “same instance, many threads” would fight Flexibility Without a Performance Tax and the cache protocol. The useful next rung is L2/L3:

1. **Document** multi-instance and BLAS nesting rules where users of C++ and `gpc` will see them.
2. **Name** process-global mutable state — especially `ndlutil`’s static Mersenne Twister — so “two models on two threads” is not a silent RNG race.
3. **Optionally harden** with `thread_local` RNG (or an explicit RNG object) and a debug owner-thread assert.

Without that, CIP-001D users and parallel experiment drivers will assume either too little (“never use threads”) or too much (“models are fine, OpenBLAS will sort itself”).

## Detailed Description

### Standards ladder (this CIP’s scope)

| Level | Meaning | This CIP |
|-------|---------|----------|
| L1 | One instance, one thread (documented) | Already done (CIP-0009); keep |
| L2 | Process contract: multi-instance rules, named globals, BLAS nesting | **In scope** |
| L3 | Make L2 true (RNG isolation or documented non-safety; optional debug assert) | **In scope** |
| L4 | Parallelism inside one evaluation | **Out of scope** → [CIP-0025](./cip0025.md) (was CIP-0013 P7) |
| L5 | Same-instance concurrent / mutexed API | **Rejected** |

### Contract text (normative intent)

Publish roughly:

1. **One live model graph, one thread.** Do not call likelihood, gradient, predict, or mutate params/data on the same `CGp` / `CGplvm` (or a kernel that holds mutable evaluation scratch shared by that model) from multiple threads.
2. **Independent instances.** Separate models with **separate** kernel (and noise) objects may be used on different threads once RNG and BLAS rules below hold. Do not share a `CKern*` that owns mutable workspaces across models used concurrently.
3. **RNG.** `ndlutil` random helpers use shared generator state. Either: (a) document as process-global — do not call `rand` / `randn` / `randperm*` concurrently — or (b) prefer `thread_local` state / an explicit RNG for new code paths (inducing init, tests). Pick (b) if cheap; otherwise (a) is enough for L2.
4. **BLAS nesting.** If the application already uses multiple threads, set the BLAS to one thread (`OPENBLAS_NUM_THREADS=1` or vendor equivalent), matching [docs/build-and-ci.md](../docs/build-and-ci.md) bench practice. Do not nest OpenBLAS threads under app threads without measurement.
5. **Python (`gpc`).** One wrapped model is not free-threaded. Keep the GIL across `optimize` unless a later change proves release-safe under this contract. Document that in `python/README.md` (and CIP-001D surface docs when they land).

### Implementation choices

**Preferred (small):**

- Expand [docs/numerics.md](../docs/numerics.md) concurrency subsection from L1 to L2/L3; link from root README and `python/README.md`.
- `thread_local` for Mersenne `mt` / `mti` in `ndlutil` **or** document process-global + “no concurrent RNG” if `thread_local` surprises MSVC / Fortran-adjacent builds — decide in implementation notes and record the choice in numerics.md.
- Optional `#ifndef NDEBUG` owner `std::thread::id` on `CGp` / `CGplvm` (set on first mutating/ensure path; assert same thread). No release mutex.

**Rejected:**

- `std::mutex` around `logLikelihood` / `updateG`.
- Removing `mutable` caches solely for threading (needs `COptimisable` const redesign; CIP-0009 deferred that).
- Folding this into CIP-0013 P7 or reopening CIP-0009.
- Requiring a multi-thread stress suite in CI (optional smoke only if cheap).

### Tenet trade-offs

- **Assert and Verify**: document invariants; optional debug thread-id assert; do not ship silent races as “undefined.”
- **Flexibility Without a Performance Tax**: keep per-instance mutable caches; no locks on the hot path.
- **Ecosystem interoperability**: Python and OpenBLAS users get the same nesting story libraries like Eigen/OpenBLAS already expect.

### Cross-links for consumers

- **CIP-001D / 001E**: cite this contract for GIL stay-held default and package README; do not re-specify C++ RNG here.
- **CIP-0013 P7**: remains a *performance* experiment; if P7 lands multi-threaded BLAS inside one evaluation, it must respect nesting rule (4) when the host is already parallel.

## Implementation Plan

1. **Docs (L2)**
   - Extend numerics.md concurrency section with the five contract bullets.
   - Pointers from README and `python/README.md`.
   - One-line cross-ref in CIP-001D docs when editing that CIP’s README surface (no scope creep into façade features).

2. **RNG (L3)**
   - Inventory call sites of `ndlutil::{rand,randn,randperm*,init_genrand}`.
   - Implement `thread_local` generator state **or** document process-global non-safety with a clear “do not race” sentence if portability blocks `thread_local`.
   - Keep `init_genrand` semantics documented (per-thread seed if `thread_local`).

3. **Optional debug assert**
   - Owner thread id on `CGp` / `CGplvm` behind existing check macros / `NDEBUG`.
   - No new dependency; abandon if noisy in legitimate move-across-threads patterns (document if abandoned).

4. **Close gate**
   - REQ-0010 acceptance criteria checked; compress contract into numerics.md (already the home) and mark CIP `compressed: true` on close.

## Backward Compatibility

- Default single-threaded CLI and tests unchanged.
- `thread_local` RNG may change cross-thread reproducibility (previously a race); same-thread seeded sequences should match.
- No `.model` stream or public API signature changes required.
- Python: documenting GIL stay-held is compatible with current CIP-001D Phase A/B bindings.

## Testing Strategy

- Existing matio / unit suites stay green (single-threaded).
- If `thread_local` RNG: a small test that two threads each `init_genrand` and draw without crashing; document that draws are independent streams, not a shared counter.
- If owner-thread assert: debug-only unit that touches a model twice on one thread (pass); optional death/assert test only if the harness already supports it — otherwise manual note.
- No requirement to race two threads on one `CGp` in CI (that remains undefined / asserted).

## Related Requirements

- [REQ-0010](../requirements/req0010_documented-concurrency-contract.md) — primary.
- Touches consumers of [REQ-000E](../requirements/req000E_gpy-facing-python-facade.md) documentation only; does not replace CIP-001D.

## Implementation Status

- [ ] Numerics + README + python README contract text
- [ ] RNG policy implemented (`thread_local` or documented process-global)
- [ ] Optional debug owner-thread assert (or explicit abandon note)
- [ ] CIP-001D/001E doc pointers (GIL / package README)
- [ ] REQ-0010 criteria satisfied; CIP Closed

## References

- [CIP-0009](./cip0009.md) — cache protocol; L1 single-threaded instance
- [docs/numerics.md](../docs/numerics.md) — current L1 note
- [CIP-0013](./cip0013.md) P7 — optional performance threading study
- [CIP-001D](./cip001D.md) / [CIP-001E](./cip001E.md) — Python façade and GIL callback notes
- OpenBLAS USAGE: multi-threaded applications should set `OPENBLAS_NUM_THREADS=1`
