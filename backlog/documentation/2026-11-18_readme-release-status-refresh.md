---
id: "2026-11-18_readme-release-status-refresh"
title: "Refresh README: drop stale 0.001 release text, align with Status section"
status: "Completed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "documentation"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- documentation
- readme
- onboarding
---

# Task: Refresh README release / status narrative

## Description

The README opens with a modern Status / badges / tutorials section, then continues into 2014-era prose that still says "Current release is 0.001". There is no tagged release, and Windows is unsupported. Newcomers get mixed signals about maturity and platform support.

## Acceptance Criteria

- [x] Remove or rewrite the "Current release is 0.001" claim
- [x] One coherent status narrative (platforms, MATLAB-free tests via matio, no Windows yet)
- [x] Keep build instructions accurate for `make.linux` / `make.osx` + libmatio

## Related

- File: `README.md`
- Docs: `docs/` maintainer notes
- Branch: `fix-silent-test-exit-readme`

## Progress Updates

### 2026-11-18

Task created from external review (Claude). Delivered on `fix-silent-test-exit-readme`: dropped the 0.001 release claim, folded platform/matio/no-tag status into the Status section, and labelled the GP-LVM 2005–2007 notes as historical.
