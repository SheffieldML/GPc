---
id: "2026-11-26_cip003d-phase2-kuu-kfu"
title: "CIP-003D phase 2: K_uu, K_fu, K_ux and kernel hooks"
status: "Completed"
priority: "High"
created: "2026-11-26"
last_updated: "2026-11-26"
category: "features"
related_cips: ["003D"]
owner: "Neil Lawrence"
dependencies: ["2026-11-26_cip003d-phase1-layout"]
tags:
- backlog
- deepgp
- weightspace
- hermite
---

# Task: CIP-003D phase 2: K_uu, K_fu, K_ux and kernel hooks

## Description

Add `wsHermiteCrossCompute` (the gradient virtuals `wsHermiteGradParams` / `wsHermiteGradX` move to phase 3, where they are gated by `checkGradients`) virtuals on `CKern` (default throw `NotImplementedError`), implement for `CMlpKern` and `CMlpardKern`, route through the compound dispatcher. Re-derive \(\rho\), \(c_k\) and the \(\sqrt{\sigma^2}\) prefactor for GPc's `CMlpKern` form (the prototype kernel has a \(2/\pi\) prefactor and a factor 2 inside; GPc's does not). Implement `computeKuu`, `computeKuf`, `computeKux`.

## Acceptance Criteria

- [x] \(K_{uu}\) and \(K_{fu}\) match numerical integration over \(\boldsymbol\omega\) at small \(D'\), several \(M\) and degrees
- [x] Convention derivation written into the CIP (so the rescaling against the prototype is explicit)
- [x] E2 raised for non-MLP kernels, naming the kernel type
- [~] Prototype \(K_{uu}\) / \(K_{fu}\) summaries compared only after rescaling; any residual difference is recorded in the CIP, not hidden by a looser tolerance

## Related

- CIP: 003D (ask-versus-delivered policy: errors E1-E6, warnings W1-W4)
- Prototype: `cip/cip0035/prototype/dgp.py`

## Progress Updates

### 2026-11-26

Task created when CIP-003D moved to Accepted.

### 2026-11-26 (completed)

`wsHermiteCrossCompute` added to `CKern` (base throws E2 naming the kernel type), `CMlpKern`, `CMlpardKern` and the compound dispatcher; `CInducingHermite::computeKuf/Kux` implemented. Derivation is in the CIP; `testInducingHermite` gates K_fu against quadrature (about 1e-11) and the erf-feature identity against `computeElement`. Prototype criterion is met analytically only (constants reconcile under x_p = x/sqrt2); the numerical comparison is deferred to phase 6, which holds the prototype reproduction.
