---
id: "2026-11-26_cip003d-phase6-reproduction"
title: "CIP-003D phase 6: reproduce the prototype composite-task row and calibrate W3"
status: "Completed"
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

- [x] Figures reproduced within a stated tolerance, or the difference and its cause recorded in the CIP. Reproduced: all four configurations fall within the prototype seed range. Root cause of the earlier gap was the KL gradient via S^-1 (plus chi applied to the bias and hex floats read as 0); see Phase 6 results in the CIP
- [x] Check whether the prototype runs at low \(d\) delivered fewer than the requested `M` windows (its allocation loop truncates silently) and note any affected reference numbers
- [x] W3 threshold set from data (0.6, from 1260-configuration sweep), replacing the 0.5 placeholder
- [x] Notebook re-run moved to `2026-11-26_cip003d-deepgp-mlp-hermite-notebook` (separate repository, needs a new notebook first)

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.

### 2026-11-26 (later)

Experiments run and recorded in `cip/cip003D.md` ("Phase 6 results"). Found and fixed a W3 defect (white noise in the prior-variance denominator) and recalibrated the threshold to 0.6. Closed 2026-11-26: gap root-caused and fixed (PR #156); results table in the CIP. Notebook re-run split into its own task.

Was open: notebook re-run (needs notebook changes in `deepgp-mlp`), and the unexplained GPc-versus-prototype gap.
