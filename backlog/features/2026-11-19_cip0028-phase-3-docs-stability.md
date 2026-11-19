---
id: "2026-11-19_cip0028-phase-3-docs-stability"
title: "CIP-0028 Phase 3: weight-space docs, compound rules, stability"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["0028"]
owner: "Neil Lawrence"
dependencies: ["2026-11-19_cip0028-phase-1-mlp-isotropic"]
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

Document compound-kernel rules for weight-space (one dot-product component), add stability cells (`ws_wv_sweep_*`), and update README / `docs/numerics.md` sparse section for points vs whitenoise vs greens vs weightspace. Compression after Close is a separate checklist item.

## Acceptance Criteria

- [ ] Compound rules enforced with clear errors
- [ ] Stability cells for \(\kappa(K_{uu})\) vs `weightVariance`
- [ ] README / `docs/numerics.md` mention weight-space inducing

## Related

- CIP: 0028
- Depends: 2026-11-19_cip0028-phase-1-mlp-isotropic

## Progress Updates

### 2026-11-19
Task created (Proposed) after CIP-0028 Accepted.
