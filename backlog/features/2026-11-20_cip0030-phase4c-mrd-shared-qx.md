---
id: "2026-11-20_cip0030-phase4c-mrd-shared-qx"
title: "CIP-0030 follow-on: MRD with shared q(X) on SVI cores"
status: "Proposed"
priority: "Low"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0030", "0017"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0030-phase4-surfaces"]
tags:
- backlog
- mrd
- svi
- bayesian-gplvm
---

# Task: CIP-0030 Phase 4c — MRD with shared \(q(\mathbf X)\)

## Description

Multi-view extension: one shared variational \(q(\mathbf X)\) with one `CSvgp` core per view (CIP-0030 Phase 4 follow-on). Collapsed `CMrd` remains unchanged.

## Acceptance Criteria

- [ ] Shared \((\boldsymbol\mu,\log\mathbf S)\) across views
- [ ] Per-view kernel / inducing / likelihood
- [ ] Stream type and CLI or demo smoke
- [ ] Gradient / ELBO checks vs independent single-view sum

## Related

- CIP: 0030

## Progress Updates

### 2026-11-20

Scoped as follow-on when Phase 4 surfaces landed.
