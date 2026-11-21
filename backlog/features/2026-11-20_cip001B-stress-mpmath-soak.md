---
id: "2026-11-20_cip001B-stress-mpmath-soak"
title: "CIP-001B: stress / mpmath soak for LQ sparse linalg"
status: "Completed"
priority: "Medium"
created: "2026-11-20"
last_updated: "2026-11-21"
category: "features"
related_cips: ["001B"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- numerics
- lq
- sparse
- soak
---

# Task: CIP-001B stress / mpmath soak

## Description

CIP-001B LQ forward/reverse and CLI default are landed. Remaining close-gate items from the CIP Implementation Status:

- Stress / mpmath soak on GP LL/grad for the large-\(N\) clustered cell
- Document that crowded-\(Z\) remains \(K_{mm}\)-limited (LQ ≡ StableFactor there) — not the Seeger/C-side regime

## Acceptance Criteria

- [x] Large-\(N\) clustered cell LL/grad soak vs StableFactor and/or mpmath reference recorded
- [x] Crowded-\(Z\) \(K_{mm}\)-limited behaviour documented in CIP-001B / numerics notes
- [x] CIP-001B ready to move Implemented → Closed after soak evidence

## Implementation Notes

Extend `numerical_stability/` / existing LQ dual-path gates; prefer documenting known \(K_{mm}\) limits over silent “LQ fixes crowded Z”.

Evidence: `bench/cip001B_soak/` (`diagLqVsStable`, `testLQAdjoint`, mpmath whitened DTCVAR with `--rtol 1e-5`). `sparse_mp_reference.py` now reads compound `white` and supports `--rtol`.

## Related

- CIP: 001B
- Docs: `docs/numerics.md`, CIP-001A stress cells
- Artifacts: `bench/cip001B_soak/README.md`

## Progress Updates

### 2026-11-20

Task created to clear VibeSafe Accepted/In Progress → backlog coverage and track post-implement soak.

### 2026-11-21

Soak delivered on branch `cip001B-stress-mpmath-soak`. Crowded-\(Z\): LQ ≡ StableFactor. Large-\(N\): dual-path + mpmath PASS. CIP-001B soak checklist closed.
