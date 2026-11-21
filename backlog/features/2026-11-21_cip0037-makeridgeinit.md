---
id: "2026-11-21_cip0037-makeridgeinit"
title: "CIP-0037: makeRidgeInit one-sided offsets and directions-first"
status: "Ready"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0037", "0028", "0035"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- weight-space
- ridge
- makeRidgeInit
---

# Task: CIP-0037 — `makeRidgeInit` layout fixes

## Description

Implement one-sided offsets and directions-before-offsets in `CInducingWeightSpace::makeRidgeInit`. Gate \(\operatorname{rank}K_{fu}=M\); keep CIP-0028 variance-captured gates. Re-record fixtures; update docs. No legacy flag.

## Acceptance Criteria

- [ ] One-sided default offsets; no \(a=0\)
- [ ] Directions-first allocation at moderate \(D'\); offsets still preferred at very small \(d\)
- [ ] \(\operatorname{rank}K_{fu}=M\) for default layout
- [ ] CIP-0028 variance-captured / \(\kappa\) gates pass
- [ ] Fixtures re-recorded; CIP-0028 / `docs/numerics.md` updated

## Related

- CIP: 0037
