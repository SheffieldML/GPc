---
id: "2026-11-15_dead-paths-comments"
title: "Remove dead paths and fix contradicting comments (CIP-000B)"
status: "Completed"
priority: "Medium"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "infrastructure"
related_cips: ["000B"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- cleanup
- maintainability
- infrastructure
---

# Task: Dead paths and contradicting comments

## Description

Accept and implement CIP-000B: inventory unreachable / commented archives and contradicting comments; fix missing throws and empty stubs; scrub stale TODOs; replace commented kernel-factory archives with CIP-000C pointers; document a short dead-code policy in `docs/workflow.md`.

## Acceptance Criteria

- [x] Inventory in CIP-000B Appendix A
- [x] `fromMxArray` throw; `pointLogLikelihood` NotImplemented; PITC typo fixes
- [x] Commented `testKern` / `CKern` factory archives → CIP-000C notes
- [x] Dead-code policy in `docs/workflow.md`
- [x] CIP-000B Implemented after local green
- [ ] CIP-000B Closed after PR CI

## Related

- CIP: 000B
- REQ: 0002
- Related CIP: 000C, 0011
- PR: https://github.com/SheffieldML/GPc/pull/36

## Progress Updates

### 2026-11-15

Task created on `cip000B-dead-paths`. Local `make USE_MATIO=1 run-matio-tests` green (0 failures). CIP-000B → Implemented; PR #36 opened.
