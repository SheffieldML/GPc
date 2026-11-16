---
id: "2026-11-16_cip0014-phase-a-path-facade"
title: "Phase A: GPy path get/set, conversions, constrain (CIP-0014)"
status: "Completed"
priority: "High"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "features"
related_cips: ["0014"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- parameters
- gpy
- features
---

# Task: CIP-0014 Phase A — path façade and constraints

## Description

Implement Phase A of [CIP-0014](../../cip/cip0014.md) / [REQ-0008](../../requirements/req0008_gpy-like-parameter-access.md): GPy-named path resolution on kernels (and noise where names exist), natural-unit get/set with documented conversions (RBF/Exp `lengthscale`), legacy GPc aliases, compound/`rbf_1` paths, dotted `getParamName`, plus `replaceTransform` / constrain-by-path.

Indexed `setParam` remains the storage API. No fix masks yet (Phase B).

## Acceptance Criteria

- [x] `paramPaths()` / `paramIndex()` / `getByPath` / `setByPath` on `CKern` with GPy canonical names
- [x] RBF `lengthscale` ↔ `inverseWidth` (\(w=1/\ell^2\)); Exp \(\ell=1/w\); `inv_lengthscale` identity alias on RBF
- [x] Legacy aliases (`inverseWidth`, `lengthScale`, `weightVariance`, …) resolve; `paramPaths()` lists GPy forms only
- [x] Compound/tensor: component aliases, `name` / `name_1` / `name_2`; dotted display names
- [x] RBF-ARD does **not** map `inputScale_i` to `lengthscale_i` (exposed as `inputScale_k`)
- [x] `replaceTransform` / `removeTransformsAt`; `constrainPositive` / `constrainZeroOne` / `constrainNone` by path
- [x] Unit tests: `testParamPaths` (paths, conversions, aliases, duplicates, constrain)
- [x] matio `testKern` 0 failures (local); `testGp` not re-run this session

## Related

- CIP: 0014
- REQ: 0008
- PR: https://github.com/SheffieldML/GPc/pull/44

## Progress Updates

### 2026-11-16

Task created (Ready) after CIP-0014 Accepted.

### 2026-11-16 (done)

Phase A landed on `cip0014-gpy-like-parameter-access`: path façade on `CKern`/`CComponentKern`, transform replace on `CTransformable`, `testParamPaths` in makefile/`matio-tests`. `./testParamPaths` and `USE_MATIO=1 ./testKern` green.
