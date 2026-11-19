---
id: "2026-11-18_cip0024-eq-cli-tests-docs"
title: "CIP-0024: CLI -k eq, tests, param-paths docs"
status: "Completed"
priority: "High"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["0024"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip0024-eq-core-factory-paths"]
tags:
- backlog
- kernels
- gpy
- cip0024
---

# Task: CLI -k eq, tests, param-paths docs

> Implements [CIP-0024](../../cip/cip0024.md); depends on core/factory/paths task.

## Description

Accept `-k eq` in CLIs that already enumerate kernel types; add EQ↔RBF and path/stream tests; document ExpQuad lengthscale-first surface in `docs/param-paths.md`.

## Acceptance Criteria

- [x] `-k eq` builds `CEqKern` in `gp` (and other CLIs that list `-k rbf`)
- [x] Help notes GPy lengthscale via `-P ExpQuad.lengthscale=…`
- [x] Tests cover \(\ell\)→\(K\) vs RBF, stream type preserved, paths
- [x] `docs/param-paths.md` has ExpQuad row

## Implementation Notes

Optional Python façade example is nice-to-have, not required to complete this task.

## Related

- CIP: 0024
- Requirement: REQ-0012

## Progress Updates

### 2026-11-18

Task created when CIP-0024 was Accepted.

### 2026-11-18
Completed on `cip0024-eq-expquad` (Python façade optional deferred).
