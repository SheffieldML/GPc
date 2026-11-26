---
id: "2026-11-26_cip003d-deepgp-mlp-hermite-notebook"
title: "CIP-003D: Hermite notebook in deepgp-mlp and DGP_FAST=0 re-run"
status: "Proposed"
priority: "Low"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_cip003d-phase6-reproduction"]
tags:
- backlog
- deepgp
- hermite
- notebooks
---

# Task: Hermite notebook in deepgp-mlp

## Description

The `deepgp-mlp` repository (`/Users/neil/lawrennd/deepgp-mlp`) has no Hermite notebook; notebooks 03 and 05 say Hermite is not exposed. Add one using `inducing='hermite'`, run it with `DGP_FAST=0`, and compare with the paper. GPc results to compare against are in the Phase 6 results of `cip/cip003D.md` (Hermite M=128 RMSE about 0.29; points about 0.22).

## Acceptance Criteria

- [ ] Hermite notebook added to `deepgp-mlp`; notebooks 03 and 05 no longer say Hermite is not exposed
- [ ] Full run (`DGP_FAST=0`) compared with the paper and the result recorded in CIP-003D

## Related

- CIP: 003D
- Separate repository: `deepgp-mlp`

## Progress Updates

### 2026-11-26

Split out of phase 6 when the rest of CIP-003D closed out.
