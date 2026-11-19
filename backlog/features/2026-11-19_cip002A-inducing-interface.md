---
id: "2026-11-19_cip002A-inducing-interface"
title: "CIP-002A follow-on: route CSvgp K_uu/K_uf through CInducing"
status: "Proposed"
priority: "Medium"
created: "2026-11-19"
last_updated: "2026-11-19"
category: "features"
related_cips: ["002A", "0028", "001C", "0029"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- svgp
- inducing
---

# Task: CIP-002A — `CSvgp` through `CInducing`

## Description

`CSvgp` currently builds \(K_{uu}\) and \(K_{uf}\) with point-inducing kernel calls (`kern.compute(Z)`). Route those matrices through the `CInducing` contract (as `CGp` already does) so SVGP can use white-noise, Green’s, and weight-space inducing without forking sparse algebra.

Spawned from CIP-0028 design review; lives under CIP-002A because it changes the SVGP model surface, not the weight-space formulas.

## Acceptance Criteria

- [ ] `CSvgp` obtains \(K_{uu}/K_{uf}\) (and grads w.r.t. inducing params) via `CInducing`
- [ ] Existing point-inducing `testSvgp` cases remain green
- [ ] At least one non-point smoke (e.g. RBF + `whitenoise`, or MLP + `weightspace` once CIP-0028 Phase 1 lands)
- [ ] Stream/CLI inducing-type selection documented or deferred explicitly

## Implementation Notes

Prefer reusing `CGp::setInducingType` patterns. Do not special-case MLP inside `CSvgp`. Weight-space needs CIP-0028 Phase 1 for a meaningful MLP smoke; WN/greens can land earlier.

## Related

- CIP: 002A (owner); 0028 / 001C / 0029 (inducing families)
- Documentation: CIP-0028 “Follow-on (not this CIP)”

## Progress Updates

### 2026-11-19
Task created (Proposed) from CIP-0028 math/design review (SVGP point).
