---
id: "2026-11-21_cip0038-phase-b-sinc-eqcosine"
title: "CIP-0038 Phase B: CSincKern + CExpQuadCosineKern + façade"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0038"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_cip0038-phase-a-wrap"]
tags:
- backlog
- kernels
- gpy
- verification
---

# Task: CIP-0038 Phase B — Sinc + ExpQuadCosine

## Description

Implement GPy-aligned `CSincKern` and `CExpQuadCosineKern` as stationary kernels with fixtures, stream/CLI, CIP-0014 paths, then expose on `gpc.kern`.

## Acceptance Criteria

- [x] Classes + `K_of_r` / grads match GPy formulas (Sinc uses `np.sinc(2r)` convention)
- [x] `testKern` fixtures green; factory type strings `sinc`, `expquadcosine`
- [x] `gpc.kern.Sinc` / `ExpQuadCosine` + parts builder
- [x] CLI `-k` accepts the new types

## Related

- CIP: 0038
