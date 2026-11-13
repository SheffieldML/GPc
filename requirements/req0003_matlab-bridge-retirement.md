---
id: "0003"
title: "MATLAB bridges can be retired without losing verification"
status: "Deferred"
priority: "Medium"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets: ["ecosystem-interoperability", "assert-and-verify"]
stakeholders: ["maintainers", "MATLAB users", "future Python/other binding users"]
tags: ["matlab", "legacy", "future-release", "testing"]
---

# REQ-0003: MATLAB bridges can be retired without losing verification

## Description

In a future release, mex / `CMatInterface` / MATLAB-linked test loading can be dropped or made optional without removing the project's ability to verify kernels, noise models, and matrices against known-good references, and without abandoning durable interchange formats (text `.model` streams, SVM-light data).

**Why this matters**: The MATLAB bridges date the stack. Retiring them is desirable, but only after REQ-0001 and REQ-0002 show the current software works, and only with a migration path for fixtures and any remaining MATLAB users.

**Who benefits**: Maintainers reducing legacy surface; users who do not have MATLAB.

## Acceptance Criteria

- [ ] A non-MATLAB verification path covers the behaviours currently gated on `.mat` fixtures (or an explicit subset is documented as retired)
- [ ] Core model load/save via text streams remains intact
- [ ] Release notes describe what was dropped and how to migrate
- [ ] Optional: a replacement language bridge (for example Python) is documented if one is chosen

## Notes (Optional)

Deferred until contemporary build and runnable verification (REQ-0001, REQ-0002) are in place. Do not delete MATLAB paths during the stabilization effort.

## References

- **Related Tenets**: ecosystem-interoperability, assert-and-verify

## Progress Updates

### 2026-11-13
Requirement created as Deferred. Near-term policy: stabilize and verify with the current MATLAB-gated suite; drop in a future release.
