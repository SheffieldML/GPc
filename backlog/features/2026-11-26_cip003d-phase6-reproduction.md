---
id: "2026-11-26_cip003d-phase6-reproduction"
title: "CIP-003D phase 6: reproduce the prototype composite-task row and calibrate W3"
status: "Ready"
priority: "Medium"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_cip003d-phase5-python"]
tags:
- backlog
- deepgp
- weightspace
- hermite
---

# Task: CIP-003D phase 6: reproduce the prototype composite-task row and calibrate W3

## Description

Run the CIP-0035 composite task (three layers, hidden width 5, \(M=64,128\)) with fixed Hermite windows through the facade and compare RMSE / NLL / ELBO per point with the table in CIP-0035 (0.546 / 0.79 / -1.01 at \(M=64\); 0.315 / 0.21 / -0.63 at \(M=128\)). Calibrate the W3 captured-variance threshold from the prototype sweeps.

## Acceptance Criteria

- [ ] Figures reproduced within a stated tolerance, or the difference and its cause recorded in the CIP
- [ ] Check whether the prototype runs at low \(d\) delivered fewer than the requested `M` windows (its allocation loop truncates silently) and note any affected reference numbers
- [ ] W3 threshold set from data, replacing the 0.5 placeholder
- [ ] Re-run the `deepgp-mlp` Hermite notebook with `DGP_FAST=0` and compare with the paper

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.
