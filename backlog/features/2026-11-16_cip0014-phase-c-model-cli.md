---
id: "2026-11-16_cip0014-phase-c-model-cli"
title: "Phase C: CGp/CLI paths and demo polish (CIP-0014)"
status: "Completed"
priority: "Low"
created: "2026-11-16"
last_updated: "2026-11-16"
category: "features"
related_cips: ["0014"]
owner: "Neil Lawrence"
dependencies: ["2026-11-16_cip0014-phase-a-path-facade"]
tags:
- backlog
- parameters
- gpy
- cli
- features
---

# Task: CIP-0014 Phase C — model paths, CLI, demos

## Description

Optional polish from [CIP-0014](../../cip/cip0014.md): model-level paths on `CGp` (`kern.*`, `beta`, `scale.*`), optional CLI `--set`, and touch demos to prefer GPy paths over raw indices. Can ship after Phase A; Phase B fix is independent.

## Acceptance Criteria

- [x] `CGp` path delegation for `kern.…` (and `beta` / `scale.j` where applicable)
- [x] Optional CLI flag to set kernel params by path (documented)
- [x] Touched demos/executables prefer `setByPath` where edited
- [x] Short path table ready for CIP closure / README compression

## Related

- CIP: 0014
- REQ: 0008

## Progress Updates

### 2026-11-16

Task created (Ready); can follow Phase A without waiting for Phase B.

### 2026-11-16 (later)

Implemented: `CGp::{paramPaths,getByPath,setByPath}`; `gp learn -P/--set path=value`; `gp.cpp` kernel init uses `setByPath`; `testCGp::testModelPaths`. Stream persistence of masks remains deferred.
