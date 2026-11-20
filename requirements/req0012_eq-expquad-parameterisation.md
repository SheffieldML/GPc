---
id: "0012"
title: "Squared-exponential covariance exposes GPy ExpQuad lengthscale-first parameterisation"
status: "Validated"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-21"
related_tenets: ["compose-through-interfaces", "ecosystem-interoperability", "assert-and-verify"]
stakeholders: ["maintainers", "modellers", "GPy users"]
tags: ["kernels", "gpy", "covariance", "parameterisation"]
---

# REQ-0012: Squared-exponential covariance exposes GPy ExpQuad lengthscale-first parameterisation

## Description

GPc exposes the squared-exponential covariance so that a GPy user can work in **ExpQuad’s parameterisation**: primary quantities are **lengthscale** \(\ell\) and **variance** \(\sigma^2\), with

\[
r=\sqrt{\sum_q\frac{(x_q-x'_q)^2}{\ell_q^2}},\qquad
k(r)=\sigma^2\exp(-\tfrac12 r^2).
\]

That surface is available under GPy-facing names **EQ** / **ExpQuad** (stream/CLI/paths). Existing `rbf` / `rbfard` remain; they may keep MATLAB-era `inverseWidth` storage. The GPy-facing contract is lengthscale-first, matching GPy `ExpQuad` (and GPy `RBF`’s default lengthscale mode), not a second covariance formula.

**Why this matters**: Interoperate With the GP Ecosystem — GPy’s clean SE teaching form is ExpQuad’s lengthscale parameterisation; RBF in GPy is the same \(K_of_r\) plus extras (psi, GPU, optional `inv_lengthscale`). GPc historically stored `inverseWidth`; CIP-0014 already converts at paths for `rbf`. This requirement makes the **ExpQuad parameterisation the explicit GPy-compat primary surface**, with naming that matches GPy. Compose Through Interfaces keeps one numerical engine; Assert and Verify requires lengthscale set/get and \(K\) to agree with that formula / with RBF at equivalent params.

**Who benefits**: GPy users porting ExpQuad scripts; façade and interchange that speak lengthscale and `GPy.kern.ExpQuad`; docs that should not force `inverseWidth` as the human API.

## Acceptance Criteria

- [x] Callers can construct / select an EQ / ExpQuad covariance whose **canonical GPy-facing parameters** are `lengthscale` and `variance` (units and meaning as in GPy `ExpQuad`)
- [x] Setting `lengthscale`=\(\ell\) yields the same \(K\) as GPy ExpQuad / as GPc RBF with `inverseWidth`=\(1/\ell^2\) within `ndlutil::MATCHTOL`
- [x] Stable stream / CLI type exists (canonical: `eq`); read aliases such as `ExpQuad` / `expquad` are accepted where types are parsed; write preserves `eq` (does not silently rewrite as `rbf`)
- [x] Parameter paths use GPy component name `ExpQuad` (e.g. `ExpQuad.lengthscale`); `inv_lengthscale` / `inverseWidth` remain available as the GPy/GPc storage-aligned aliases where already defined for RBF
- [x] Non-goals are explicit: no second numerical `K_of_r` implementation; no requirement to change `rbf` indexed/`setParam` storage away from `inverseWidth`; no `ExpQuadCosine` / `EQ_ODE*`; no deprecation of `rbf` / `rbfard`; no `eqard` in the first delivery

## Notes (Optional)

In current GPy, `ExpQuad` and `RBF` are **siblings** under `Stationary` (same \(K_of_r\)); RBF is not a subclass of ExpQuad. Pedagogically ExpQuad is the lengthscale-primary SE; RBF adds variational/psi machinery and an optional `inv_lengthscale` mode. CIP-000C deferred ExpQuad as duplicate *maths*; this requirement is **parameterisation and naming at the GPy boundary**.

Internal storage may still use `inverseWidth` for a shared engine with `CRbfKern` (conversion at the path / GPy façade). Indexed `setParam` for `rbf` need not change.

**Numbering note**: REQ-0011 on the CSV/pods branch is a different requirement; this EQ requirement is **0012**.

## References

- **Related Tenets**: compose-through-interfaces, ecosystem-interoperability, assert-and-verify
- **External Links**: GPy `GPy.kern.ExpQuad`, `GPy.kern.RBF`, `Stationary` lengthscale scaling
- **Related**: [REQ-0005](./req0005_gpy-aligned-kernel-coverage.md), [REQ-0008](./req0008_gpy-like-parameter-access.md), [REQ-000A](./req000A_shared-model-interchange.md)
- **HOW**: [CIP-0024](../cip/cip0024.md)

## Progress Updates

### 2026-11-18
Drafted as REQ-0011 alongside an early CIP-0022 sketch; renumbered to **REQ-0012** / **CIP-0024** after CIP-0022+REQ-0011 were claimed by CSV/pods work, and CIP-0023 by PPA renumber. Centred on lengthscale-first ExpQuad parameterisation. Moved to **Ready** when CIP-0024 was Accepted.

### 2026-11-18 (later)
PPA integrated on main; CIP-0024 → In Progress on `cip0024-eq-expquad`.

### 2026-11-21
CIP-0024 Closed. Status → Validated.
