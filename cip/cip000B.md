---
author: "Neil Lawrence"
created: "2026-11-13"
id: "000B"
last_updated: "2026-11-20"
status: "Closed"
compressed: false
related_requirements: ["0002"]
related_cips: ["0005", "0008", "000C", "0011"]
tags:
- cip
- cleanup
- maintainability
- infrastructure
title: "Remove dead paths and fix contradicting comments"
---

# CIP-000B: Remove dead paths and fix contradicting comments

## Status

- [x] Proposed
- [x] Accepted
- [x] In Progress
- [x] Implemented
- [x] Closed

## Summary

Delete or finish unfinished code that cannot run, and correct comments that contradict the implementation, so reviewers are not misled and coverage reports stop counting unreachable blocks as intentional surface.

## Motivation

GPc retains experimental or superseded fragments: code after a `throw NotImplementedError` (for example in `CGp::updateAD`), large commented-out tests that were restored elsewhere ([CIP-0005](cip0005.md)), and comments that still disagree with behaviour (historical 380/513 labelling was one instance). Leaving dead paths in place increases review cost and hides real gaps in coverage.

This is lower risk than [CIP-0008](cip0008.md)–[CIP-000A](cip000A.md) and can proceed in parallel once those owners agree on sequencing. It supports **Assert Invariants and Verify Against Reference** by making the executable surface match the documented one. It also unblocks clearer coverage accounting for [CIP-0011](cip0011.md).

## Detailed Description

1. **Unreachable code** — Find `throw` / `return` / `#if 0` blocks that leave trailing statements; delete the dead tail or implement the missing path with tests.
2. **Commented archives** — Remove large commented blocks that duplicate history already in git, or move short rationale into a CIP/docs note. Do not delete without checking that CIP-0005 already restored the needed tests.
3. **Comment audit** — Fix labels and parameter descriptions that contradict Fortran/C++ (remaining ACM or BLAS comments). Prefer short accurate comments over essays.
4. **Optional** — Note candidates for splitting `CKern.cpp` (~4,400 lines) into per-family files; implement the split only if this CIP stays small enough, otherwise spawn a follow-on CIP.

### Non-goals

- Behavioural changes to working sparse GP / GPLVM paths.
- Ownership or cache redesign (0008–000A).
- Implementing `CKern::getGradParam` stubs (intentional `NotImplementedError`; cover under CIP-0011 / kernel CIPs).

## Implementation Plan

1. Grep-driven inventory of `NotImplemented`, `#if 0`, and obvious dead tails; attach the list to this CIP.
2. Remove or implement each item with a one-line rationale in the commit message (`CIP-000B: ...`).
3. Confirm Reference and Compile CI remain green.
4. Compress a short “dead code policy” sentence into [docs/workflow.md](../docs/workflow.md) when closed.

## Implementation Status

- [x] Inventory attached (Appendix A)
- [x] Fix missing `throw` in `CGp::fromMxArray`
- [x] `pointLogLikelihood` → `NotImplementedError` (was empty body)
- [x] Fix PITC typo (`PTIC` / `implemnted`)
- [x] Remove stale “inner products” / maxTries TODOs that contradicted the code
- [x] Replace commented kernel-factory / `testKern` archives with CIP-000C pointers
- [x] Scrub deferred `testNoise` / `testIvm` commented stubs
- [x] Dead-code policy sentence in `docs/workflow.md`
- [x] CI green; CIP Closed

## Testing Strategy

- Compile + Reference tests; no new fixtures unless a path is implemented rather than deleted.

## Related

- Review note: unfinished code and contradicting comments (2026-11-13)
- [CIP-0005](cip0005.md) ACM / transpose cleanup
- [CIP-000C](cip000C.md) additional kernel types (periodic / Gibbs / ARD Matérn)
- [CIP-0011](cip0011.md) CGp coverage (expect-throw stubs)

## Appendix A — Inventory (2026-11-15)

| Item | Disposition |
| --- | --- |
| `CGp::fromMxArray` missing `throw` before `NotImplementedError` | Fixed (would have fallen through) |
| `CGp::pointLogLikelihood` empty body | Throws `NotImplementedError` |
| Commented FITC alt path after `_multiplyInvKuu` | Already gone on main post CIP-000E |
| `updateAD` “TODO generate/update the inner products” | Removed (stale) |
| `maxTries` “isn't being properly used” comment | Corrected |
| “PTIC” / “implemnted” in X-opt error | Fixed spelling |
| Speculative storage / beta / compute TODOs in `CGp` | Rephrased as factual notes |
| Commented Gibbs/periodic/`*ard` stubs in `testKern` / `CKern` stream factory | Removed; pointer to CIP-000C |
| Commented `mgaussian`/`cmpnd` in `testNoise`; ordered in `testIvm` | One-line deferral notes |
| Intentional `NotImplementedError` (non-spherical, back constraints, `outGrad*`, `getGradParam`) | **Kept** — real API surface for CIP-0011 |
| `#if 0` blocks | None found |
| `CKern.cpp` split | Deferred (optional; not in this PR) |
