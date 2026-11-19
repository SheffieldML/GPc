---
id: "2026-11-19_cip0028-phase-3-docs-stability"
title: "CIP-0028 Phase 3: weight-space docs, compound rules, stability"
status: "Completed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0028-phase-1-ridge"]
tags:
- backlog
- sparse
- inducing
- weight-space
- numerics
- docs
---

# Task: CIP-0028 Phase 3 — docs, compound rules, stability

## Description

Document compound-kernel rules for weight-space (one dot-product component; align with CIP-0027 defaults), add stability cells focused on **variance captured** / sparse–exact gap under a \(\kappa(K_{uu})\) budget, and update README / `docs/numerics.md` for points vs whitenoise vs greens vs **ridge** weightspace. Compression after Close is a separate checklist item. Do not start until ridge Phase 1 lands.

## Acceptance Criteria

- [x] Compound rules enforced with clear errors
- [x] Stability cells: primary metric variance-captured / sparse bound vs exact under \(\kappa(K_{uu})\le10^4\); finiteness; \(\kappa\) flat-in-`weightVariance` demoted
- [x] README / `docs/numerics.md` mention ridge weight-space inducing; MLP (finite \(\Sigma\)) vs arccos-0 (infinite / angular)
- [x] Optional: note / sketch comparison vs Dutordoir activated features (conditioning) — full GPflow bake-off can be a follow-on

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-1-ridge

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted.

### 2026-11-19 (ridge)
Blocked on ridge Phase 1; gates updated to variance-captured.

### 2026-11-19 (done)
`CCmpndKern` weight-space path selects one `mlp|mlpard|arccos` (+ optional silent `white`); Bias/RBF/second signal throw clear Errors. Stability cell `ws_ridge_wv_sweep_*` (κ flat in wv, capture ≥0.25, StableFactor/LQ finite). README + `docs/numerics.md` updated for ridge weightspace.
