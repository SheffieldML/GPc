---
id: "0004"
title: "Codebase compiles under a modern C++ standard without legacy pins"
status: "Validated"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets: ["flexibility-with-performance", "assert-and-verify"]
stakeholders: ["maintainers"]
tags: ["cxx", "toolchain", "build", "modernization"]
---

# REQ-0004: Codebase compiles under a modern C++ standard without legacy pins

## Description

The core tools and (once restored) reference tests build with a contemporary C++ language mode used by current compilers by default or by explicit project choice of C++17 or newer — without requiring `-std=c++14` solely to tolerate removed language features such as dynamic exception specifications.

**Why this matters**: CIP-0001 pinned `-std=c++14` so CI could go green on Ubuntu’s g++ 15. That pin is a bridge, not the destination. Flexibility Without a Performance Tax assumes we can keep building on the toolchains people actually install.

**Who benefits**: Maintainers and CI; anyone building from source on current Linux/macOS toolchains.

## Acceptance Criteria

- [x] Project compile flags target C++17 or newer (documented in make fragments / CI)
- [x] No dynamic exception specifications (`throw(Type)`) remain in headers or sources that CI compiles
- [x] Compile CI remains green without a C++14-only pin
- [x] Known iostream / API mismatches introduced by modern libstdc++ (and cousins) that block the core tools are fixed, not papered over with an older `-std`

## Notes (Optional)

Scope is language-mode and compile fixes needed for that mode. Broader style modernization (smart pointers everywhere, etc.) is optional and should not block this requirement.

## References

- **Related Tenets**: flexibility-with-performance, assert-and-verify
- **Related**: CIP-0001 temporary `-std=c++14` pin in `make.linux`

## Progress Updates

### 2026-11-13
Requirement created after compile CI landed with a C++14 bridge.

Implemented on `cip0002-cxx17-modernization` / PR #14; Compile CI green under `-std=c++17`.
