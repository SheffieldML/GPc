---
id: "2026-11-20_cip0030-phase2-inducing-families"
title: "CIP-0030 Phase 2: inducing families on CSviBayesGplvm"
status: "Completed"
priority: "High"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "002A", "001C", "0028", "0029"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0030-phase1-svi-bayes-gplvm"]
tags:
- backlog
- svgp
- svi
- bayesian-gplvm
- inducing
- weight-space
---

# Task: CIP-0030 Phase 2 — white-noise, weight-space, Green's on `CSviBayesGplvm`

## Description

Wire all `CInducing` families through `CSviBayesGplvm::setInducingType` and verify gradients / end-to-end smoke for points, white-noise RBF, weight-space ridge (MLP erf and arc-cosine soft-ReLU), and Green's (1-D Brownian).

## Acceptance Criteria

- [x] `setInducingType` delegates to `CSvgp` (points / whitenoise / weightspace / greens)
- [x] V4: FD at fixed ε for \(\boldsymbol\mu,\log\mathbf S\) and global (kern / inducing / \(\beta\)) params on each family
- [x] V8: demo-sized RBF/points MC ELBO matches collapsed at \(q(\mathbf u)^*\); weight-space MLP short train + \(K=1000\) estimate finite
- [x] `testSviBayesGplvm` green (`Number of failures: 0`)

## Implementation Notes

No family-specific code in the model — composition via `CInducing` only. Soft-ReLU is `CArcCosineKern` order 1 + `weightspace`.

## Related

- CIP: 0030

## Progress Updates

### 2026-11-20

Phase 2 implemented and verified.
