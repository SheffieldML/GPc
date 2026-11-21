---
id: "2026-11-21_weightspace-hermite-ridges"
title: "Hermite ridge windows as CIP-0028 follow-on (stable K_uu)"
status: "Proposed"
priority: "Medium"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0028", "0032"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- weightspace
- numerics
- inducing
---

# Task: Hermite ridge windows (optional inducing family)

## Description

Gaussian ridge windows in whitened weight space ([CIP-0028](../../cip/cip0028.md)) have a clean “finite MLP” mean but \(K_{uu}\) conditioning collapses when \(s\) grows vs offset spacing, at \(s=\sqrt2\), or with \(a=0\). Guards in `CInducingWeightSpace` (bounds on \(s\), no zero offsets) mitigate that for the Gaussian family.

**Hermite ridges** \(\psi_{v,k}(\boldsymbol\omega)=\mathrm{He}_k(\mathbf v^\top\boldsymbol\omega)/\sqrt{k!}\,\sqrt{\mathcal N(\boldsymbol\omega;0,I)}\) (odd \(k\)) give closed-form \(K_{uu}=\delta_{kl}(\mathbf v^\top\mathbf v')^k\) and \(K_{fu}\) polynomial in \(\rho_v\), with \(\kappa\) typically tens–hundreds and often higher variance capture in prototype sweeps — at the cost of losing the smoothed-erf neuron interpretation.

## Acceptance Criteria

- [ ] Separate CIP drafted (optional inducing type alongside Gaussian ridge)
- [ ] Closed forms + MC/quadrature gates; capacity rule per degree
- [ ] Document trade-off vs Gaussian ridges (large \(\|\hat{\mathbf x}\|\), no MLP mean story)

## Related

- CIP: 0028, 0032
- Prototype notes from 2026-11-21 stability review (Hermite Mehler / arcsine Taylor)

## Progress Updates

### 2026-11-21

Task created after ridge \(K_{uu}\) stability diagnosis; Gaussian guards land with CIP-0032 follow-on commits.
