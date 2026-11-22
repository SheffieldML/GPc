---
id: "2026-11-22_cip0035-shared-layer-covariance"
title: "CIP-0035: optional shared q(v) covariance across layer outputs"
status: "Proposed"
priority: "Low"
created: "2026-11-22"
last_updated: "2026-11-22"
category: "features"
related_cips: ["0035"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- deep-gp
- svgp
- performance
---

# Task: Shared layer covariance for DeepGP / CSvgp

## Description

[CIP-0035](../../cip/cip0035.md) documents an optional matrix-normal \(q\): one shared \(\tilde S\) across a layer's \(D_\ell\) outputs, cutting covariance cost and parameters by \(D_\ell\). Default remains per-output (current `CSvgp`). The JAX prototype shares one covariance across the hidden width (5), so DeepGP timing comparisons are slightly unfair to GPc on covariance work.

## Acceptance Criteria

- [ ] Opt-in shared-\(\tilde S\) mode on a layer (API + stream flag)
- [ ] ELBO / grads FD-green against the independent-output model when outputs are exchangeable, or documented divergence
- [ ] Document when to use shared vs per-output in the DeepGP tutorial

## Implementation Notes

- Design already in CIP-0035 “Shared covariance option”; not required for correctness.
- Lower priority than MLP bulk grads and KL-from-chol.

## Related

- CIP: 0035

## Progress Updates

### 2026-11-22

Task created from DeepGP vs JAX timing notes (prototype shares \(q\) covariance across hidden outputs).
