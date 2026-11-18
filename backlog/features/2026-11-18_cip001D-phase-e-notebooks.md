---
id: "2026-11-18_cip001D-phase-e-notebooks"
title: "CIP-001D Phase E: Jupyter examples, pods, thin plot extras"
status: "Ready"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
category: "features"
related_cips: ["001D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-18_cip001D-phase-b-gp"]
tags:
- backlog
- python
- jupyter
- pods
- plotting
- examples
---

# Task: CIP-001D Phase E notebooks and extras

## Description

Ship GPy-familiar **examples** without bloating core `gpc`: Jupyter notebooks using external `pods` for datasets, matplotlib figures, and optional thin `plot_1d` under `gpc[plot]` / `gpc[examples]`.

## Acceptance Criteria

- [ ] At least one toy 1-D notebook and one `pods` dataset notebook (e.g. olympic marathon) that import core `gpc` only for modelling
- [ ] `pods` / matplotlib / jupyter are optional extras — not required to `import gpc` or run Phase A/B pytest
- [ ] Optional thin 1-D mean±2σ helper documented; no full GPy plot surface
- [ ] Smoke recipes use noisy observations (document noiseless white→0 behaviour)
- [ ] CIP-0015 remains native-first share format

## Implementation Notes

Mirror GPy: examples `try: import pods` / tell user to `pip install pods`. Prefer `python/examples/*.ipynb` or `docs/notebooks/`.

## Related

- CIP: 001D
- Related data I/O (native CSV + pods preference): [CIP-0022](../../cip/cip0022.md) / [REQ-0011](../../requirements/req0011_csv-pods-data-access.md)
- External: [pods](https://pypi.org/project/pods/), GPy `examples/regression.py`

## Progress Updates

### 2026-11-18

Task created after agreeing core vs extras vs notebooks layering.
