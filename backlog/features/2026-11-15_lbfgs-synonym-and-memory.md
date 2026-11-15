---
id: "2026-11-15_lbfgs-synonym-and-memory"
title: "Accept lbfgs as quasinew synonym and fix L-BFGS buffer lifetime"
status: "Completed"
priority: "Medium"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "features"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- optimiser
- lbfgs
---

# Task: L-BFGS synonym and buffer lifetime

## Description

Make `-O lbfgs` a synonym for Netlab-style `quasinew`, and stop leaking the L-BFGS work buffers in `COptimisable::lbfgsOptimise`. While touching that loop, fix the reverse-communication success path so `iflag==0` terminates instead of falling through.

## Acceptance Criteria

- [x] `setDefaultOptimiserStr` and CLI parsers (`gp`, `gplvm`, `ivm`) accept `lbfgs` as well as `quasinew`
- [x] Help text documents the synonym
- [x] L-BFGS scratch storage is owned by `vector<>` (freed on return and throw)
- [x] Successful L-BFGS termination (`iflag==0`) exits the loop and syncs parameters
- [x] Rebuild with blessed `make.osx` → `make.inc` and smoke-run `./gp learn -O lbfgs ...`

## Implementation Notes

- Canonical stream / MATLAB name remains `quasinew` via `getDefaultOptimiserStr`
- Enum value is still `BFGS`; it already called `lbfgsOptimise()` (Nocedal Fortran in `ndlfortran.f`)
- Local build uses the CIP-0001 Homebrew `make.osx` already on `main` (macOS CI matrix from CIP-0006)

## Related

- CIP: none (small optimiser polish)
- Build: `make.osx` / CIP-0001, CIP-0006

## Progress Updates

### 2026-11-15

Re-applied on current `main` after fast-forward (local tree had been behind the macOS make/CI merges). Smoke: `./gp learn -O lbfgs -# 5 examples/sinc.svml` → L-BFGS `IFLAG = 0`, reported `Optimiser: quasinew`.

### 2026-11-15 (re-land)

Changes were lost before the first commit; re-applied on `fix-lbfgs-synonym-memory` from `origin/main` (post CIP-0008 merge) for PR.

