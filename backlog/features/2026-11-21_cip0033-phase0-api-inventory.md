---
id: "2026-11-21_cip0033-phase0-api-inventory"
title: "CIP-0033 Phase 0: ep_moments API freeze + CNoise site inventory"
status: "Completed"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0033", "0027"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- ep
- laplace
- likelihood
- inference
---

# Task: CIP-0033 Phase 0 — API freeze and inventory

## Description

Before coding EP/Laplace: freeze batched likelihood signatures (especially `ep_moments`) and catalogue every use of `CNoise` data / site fields (`py`, `mu`, `varSigma`, ADF helpers) across IVM, PPA, `CGp`, and the Python façade. Lock FTC-only vs sparse EP for v1 at Accept.

## Acceptance Criteria

- [ ] Signature table checked into `cip/cip0033/` (or Phase 0 note)
- [ ] Inventory spreadsheet / markdown of site-touch call sites
- [ ] Accept decision recorded on CIP-0033: FTC-first EP (recommended) or sparse-in-v1
- [ ] CIP-0033 status → Accepted when ready

## Related

- CIP: 0033
- REQ: 0019

## Progress Updates

### 2026-11-21

Task created with CIP-0033 / REQ-0019 Proposed.

### 2026-11-21 (done)

Accept locks + inventory + GPy mirror note landed; `ep_moments` implemented. Status → Completed.
