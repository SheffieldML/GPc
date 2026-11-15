---
id: "2026-11-15_fitc-logtwopi-double-count"
title: "FITC log-likelihood double-counts N log 2π"
status: "Proposed"
priority: "High"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "bugs"
related_cips: ["0012"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- bugs
- fitc
- likelihood
- numerics
---

# Task: FITC 2π double count in log-likelihood

## Description

In `CGp::logLikelihood`, the FITC branch adds `N * LOGTWOPI` before `L *= -0.5`. The common tail then subtracts `d * N * HALFLOGTWOPI` again. The reported FITC log-likelihood is therefore low by exactly \(\tfrac{d N}{2}\log 2\pi\) in both Legacy and StableFactor modes.

Observed in CIP-000E stress work (2026-11-15): for \(N{=}200\), \(d{=}1\), the offset is 183.7877, matching the analytic constant to 1e-11 against an mpmath reference. **Gradients are unaffected.**

Tracked out of [CIP-0012](../../cip/cip0012.md) (F5) so the whitened-path CIP stays focused on factorisation.

## Acceptance Criteria

- [ ] FITC branch no longer double-counts the \(2\pi\) term (match DTC/DTCVAR/PITC convention)
- [ ] `testGp` FITC arm still green (update fixture and/or the existing \(\pm N/2\log 2\pi\) tolerance helper if needed)
- [ ] Note in `docs/numerics.md`; CIP-0012 F5 item can be marked fixed

## Related

- CIP: 0012
- Code: `CGp.cpp` FITC `logLikelihood` case + common tail
