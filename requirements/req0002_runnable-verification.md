---
id: "0002"
title: "Reference verification suite is runnable"
status: "Ready"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets: ["assert-and-verify", "ecosystem-interoperability"]
stakeholders: ["maintainers"]
tags: ["testing", "verification", "stabilization"]
---

# REQ-0002: Reference verification suite is runnable

## Description

The in-tree reference tests (`testKern`, `testNoise`, `testMatrix`, and the other `tests` makefile targets) can be built and executed on the same contemporary platform used for REQ-0001, and their pass/fail result is trustworthy enough to gate changes.

It is acceptable in this requirement that verification still depends on MATLAB-era `.mat` fixtures and `_NDLMATLAB` if that is what makes the current suite work. Removing that dependency is explicitly out of scope here and belongs to a later release (see REQ-0003).

**Why this matters**: Assert Invariants and Verify Against Reference is empty if the only verification path cannot be run. Today the sources and `matfiles/` fixtures exist, but there is no working default test path and no CI.

**Who benefits**: Maintainers changing kernels, noise models, matrices, or build flags.

## Acceptance Criteria

- [ ] Documented steps build the `tests` targets (or an equivalent documented subset) on the contemporary platform
- [ ] Running that suite reports clear pass/fail against `matfiles/` (or successor fixtures)
- [ ] At least the kernel and matrix reference tests are exercised successfully after a clean build
- [ ] The README or equivalent maintainer notes state how to run verification

## Notes (Optional)

Non-goals for this requirement: introducing a new test framework, replacing `.mat` fixtures, or adding CI cloud runners (CI may follow once local verification works).

## References

- **Related Tenets**: assert-and-verify, ecosystem-interoperability
- **Related**: REQ-0003 (future MATLAB bridge retirement)

## Progress Updates

### 2026-11-13
Requirement created. Makefile comment: "To compile tests, the MATLAB interface must be enabled (i.e. define `_NDLMATLAB`)". No GitHub Actions workflows present.
