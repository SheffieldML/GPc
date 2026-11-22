---
id: "2026-11-22_cip003A-phase5-ve-docs"
title: "CIP-003A Phase 5: SVGP/DeepGP VE + generalised Laplace + docs"
status: "Completed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
owner: ""
dependencies: ["2026-11-22_cip003A-phase4-python-facade"]
---

# Task: CIP-003A Phase 5 VE hooks, generalised Laplace, docs

## Description

Wire Gauss–Hermite `expectedLogLikelihood` verification, generalise
`CLaplaceInference` over CIP-003A `CNoise` families, expose
`GPRegression(..., likelihood=, inference='laplace')`, and update the subset
checklist / SVGP / likelihood-zoo tutorials.

## Acceptance Criteria

- [x] VE FD checks for at least Poisson and Student-t
- [x] `CLaplaceInference::setNoise` + `ndlnoise::laplacePointDerivs`
- [x] `CGp` Student-t Laplace smoke in `testLikelihoodZoo`
- [x] Python Laplace `GPRegression` smoke
- [x] Docs: `svgp.md`, `subset_checklist.md`, `likelihood_zoo.md`

## Related

- CIP: 003A

## Progress Updates

### 2026-11-22
Phase 5 implemented on `cip003A-likelihood-zoo`.
