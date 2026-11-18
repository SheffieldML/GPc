---
id: "2026-11-18_cip001C-stability-cells"
title: "CIP-001C Phase 4: stability-suite point vs white-noise cells"
status: "Ready"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001C", "0019", "001A"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- sparse
- inducing
- numerics
- stability
---

# Task: CIP-001C Phase 4 — stability suite cells

## Description

Add numerical-stability suite cells (CIP-0019/001A) comparing point vs white-noise inducing on a lengthscale sweep with fixed \(Z\): finiteness, jitter added, \(\kappa(K_{uu})\), ELBO gap, under `stable` and `lq`.

## Acceptance Criteria

- [ ] Lengthscale-sweep cell in `numerical_stability/`
- [ ] White-noise path finite with no \(K_{uu}\) jitter across the sweep
- [ ] Point path jitter/failure recorded as expected contrast (xfail-style where it breaks)
- [ ] ELBO/bound gap vs (jittered) point inducing recorded
- [ ] Metrics emitted in CIP-001A categorised format where applicable

## Related

- CIP: 001C
- REQ: 000D
