---
id: "2026-11-13_rename-default-branch-main"
title: "Rename default branch master to main"
status: "In Progress"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "infrastructure"
related_cips: ["0001"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- git
- infrastructure
---

# Task: Rename default branch to main

## Description

Rename the repository default branch from `master` to `main` (local and GitHub), and ensure clones/CI track `main`.

## Acceptance Criteria

- [ ] Local branch renamed to `main`
- [ ] Remote `main` exists and is the GitHub default branch
- [ ] CI workflows trigger on `main`
- [ ] `master` removed or clearly non-default after cutover

## Implementation Notes

Use `git branch -m`, push `-u origin main`, then `gh repo edit --default-branch main` (requires admin + valid `gh` auth).

## Related

- CIP: 0001

## Progress Updates

### 2026-11-13

Task created with CIP-0001 acceptance. Starting cutover.
