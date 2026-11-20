---
id: "2026-11-19_cip002A-inducing-interface"
title: "CIP-0030 Phase 0a / CIP-002A: route CSvgp K_uu/K_uf through CInducing"
status: "Completed"
priority: "High"
created: "2026-11-19"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "002A", "0028", "001C", "0029"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- svgp
- inducing
---

# Task: CIP-0030 Phase 0a — `CSvgp` through `CInducing`

## Description

`CSvgp` currently builds \(K_{uu}\) and \(K_{uf}\) with point-inducing kernel calls (`kern.compute(Z)`), and only routes gradients through a temporary `CInducingPoints`. Own a `CInducing*` (as `CGp` does) so SVGP can use white-noise, Green’s, and weight-space inducing without forking sparse algebra.

Spawned from CIP-0028 design review under CIP-002A; **delivery owned by CIP-0030 Phase 0a** (independently shippable).

## Acceptance Criteria

- [x] `CSvgp` obtains \(K_{uu}/K_{uf}\) (and grads w.r.t. inducing params) via owned `CInducing*`
- [x] Existing `Z` constructor wraps `CInducingPoints`; `setInducing(CInducing*)` takes ownership
- [x] `getOptNumParams` / `getOptParams` / `setOptParams` pack inducing params via `CInducing::getNumParams`
- [x] Existing point-inducing `testSvgp` cases remain green
- [x] At least one non-point smoke (RBF + `whitenoise`)
- [ ] Stream/CLI inducing-type selection deferred to CIP-0030 Phase 4 / CIP-002A Phase 5

## Implementation Notes

Prefer reusing `CGp::setInducing` ownership patterns. Do not special-case MLP inside `CSvgp`. Keep member centres storage (`Z`) for families that need a non-owned `CMatrix*` (points, white-noise, Green's); weight-space owns `W` itself.

## Related

- CIP: 0030 (delivery); 002A / 0028 / 001C / 0029 (inducing families)

## Progress Updates

### 2026-11-19
Task created (Proposed) from CIP-0028 math/design review (SVGP point).

### 2026-11-20
Retargeted to CIP-0030 Phase 0a; priority High; status In Progress. Implementation started on branch `cip0030-svi-bayes-gplvm`.

### 2026-11-20 (later)
Phase 0a core landed: owned `CInducing*`, `setInducing` / `setInducingType`, matrices and grads via inducing. `testSvgp` green including white-noise smoke. CLI/stream inducing-type selection still deferred.

### 2026-11-20 (Phase 0b)
Phase 0a acceptance criteria for the model core are met; remaining CLI/stream inducing-type flag stays deferred to Phase 4. Status Completed for the inducing-routing deliverable.

