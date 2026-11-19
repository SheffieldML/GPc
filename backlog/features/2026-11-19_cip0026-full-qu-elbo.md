---
id: "2026-11-19_cip0026-full-qu-elbo"
title: "CIP-0026: full q(u) PPA ELBO matching dense Cov terms"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0026"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0026-phase-c-dtcvar-bound"]
tags:
- backlog
- ppa
- elbo
- dtcvar
---

# Task: CIP-0026 full \(q(u)\) PPA ELBO

> Backlog for [CIP-0026](../../cip/cip0026.md) optional follow-up (now done).

## Description

Replace the interim sparse monitor with the inducing-form PPA ELBO:
\(-\mathrm{KL}(q(u)\|p(u)) - \tfrac12\beta\,\mathrm{tr}(\mathrm{Cov}_{\bar f})\) (+ DTCVAR \(K-Q\)) + kern prior + outer noise LL.
Anchor \(q(u)\) to `fPost` (the \(\langle f\rangle\) that produced `fBar`).

## Acceptance Criteria

- [x] `CGp::sparsePpaElboGpTerms()` for DTC/DTCVAR
- [x] `CSparsePpa::ppaBound()` uses full ELBO; FITC/PITC keep interim
- [x] DTC/DTCVAR \(M=N\) matches dense `approxLogLikelihood` (~1e-14)
- [x] \(M<N\) finite; no \(N\times N\) `invK` / `fBarfBar` on the sparse path

## Related

- CIP: 0026

## Progress Updates

### 2026-11-19

Landed and gated in `testSparsePpa`.
