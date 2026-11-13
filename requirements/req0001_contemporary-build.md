---
id: "0001"
title: "Core tools build on a contemporary platform"
status: "Ready"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets: ["flexibility-with-performance", "ecosystem-interoperability"]
stakeholders: ["maintainers", "users building from source"]
tags: ["build", "platform", "stabilization"]
---

# REQ-0001: Core tools build on a contemporary platform

## Description

A maintainer or user can build the core command-line tools (`gp`, and where still supported `ivm` / `gplvm`) from this repository on at least one currently maintained platform (for local work: modern macOS with a documented BLAS/LAPACK path), using instructions that match reality.

**Why this matters**: Flexibility Without a Performance Tax only holds if the C++ tools still compile. Interoperate With the GP Ecosystem requires meeting users on platforms they actually use today, not only 2005-era Linux atlas layouts.

**Who benefits**: Maintainers verifying changes; anyone following the README examples.

## Acceptance Criteria

- [ ] Documented build steps succeed on at least one contemporary developer platform without undocumented hacks
- [ ] `gp` (and any other tools claimed by the README as buildable) link successfully against a working BLAS/LAPACK
- [ ] Architecture/make configuration for that platform is checked in or clearly named (no silent reliance on a mismatched `make.linux` include)
- [ ] A short smoke command from the README (for example help or a tiny learn) runs after the build

## Notes (Optional)

Near-term scope is restore and document what works. Retiring unused executables or platforms can wait for a later release.

## References

- **Related Tenets**: flexibility-with-performance, ecosystem-interoperability

## Progress Updates

### 2026-11-13
Requirement created. Makefile currently hard-includes `make.linux`; macOS/macports make fragments look stale; no CI.
