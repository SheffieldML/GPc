---
id: "2026-11-20_cip0027-phase-bp-ivm-softener"
title: "CIP-0027 Phase B′: IVM composition tests + keep-White exception"
status: "Completed"
priority: "Medium"
created: "2026-11-20"
last_updated: "2026-11-20"
category: "features"
related_cips: ["0027"]
owner: "Neil Lawrence"
dependencies: ["2026-11-20_cip0027-phase-b-default-flip"]
tags:
- backlog
- ivm
- white
- softener
---

# Task: CIP-0027 Phase B′ — IVM softener evidence

## Description

Run the composition matrix in [`cip/cip0027/ivm_softener.md`](../../cip/cip0027/ivm_softener.md). Expected outcome: keep auto White for IVM; decide Bias; write the exception into IVM tutorials and CLI help. Optional spike: trainable probit σ² as evidence only for the follow-on CIP.

## Acceptance Criteria

- [x] Composition results written to `cip/cip0027/ivm_kernel_composition.md`
- [x] IVM tutorials / CLI help document keep-White exception
- [x] Bias auto-append decision recorded (drop or keep) with rationale
- [x] Optional probit-σ² spike (if done) filed as evidence, not shipped default

## Related

- CIP: 0027
- Documentation: `cip/cip0027/ivm_softener.md`, `cip/cip0027/ivm_kernel_composition.md`

## Progress Updates

### 2026-11-20

Task created on Accept.

### 2026-11-20 (done)

`diagIvmSoftener`: keep White (Jaccard RBF-only 0.32); drop auto Bias (RBF+White Jaccard 0.92). Docs + `ivm.cpp` updated. Probit-σ² spike deferred.
