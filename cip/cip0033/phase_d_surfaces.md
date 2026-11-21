# CIP-0033 Phase D — strip / surfaces (2026-11-21)

## FTC-only lock (reaffirmed)

Dense `CGp` EP and Laplace are the v1 product path. **Sparse EP / GPy `EPDTC` remain out of scope** until a later CIP. IVM + White softener stays the large-\(N\) active-set route.

## Stream note

`CNoise::writeParamsToStream` / `readParamsFromStream` already serialise **parameters only** (type, outputDim, param vector). They do **not** stream `py`, `mu`, or `varSigma`. Phase D does not bump the stream version.

## Site workspace dual-write

- New type: [`CNoiseSiteWorkspace.h`](../../CNoiseSiteWorkspace.h) (`mu`, `varSigma`).
- `CNoise::attachSiteWorkspace` + bulk `setMus` / `setVarSigmas` dual-write when attached.
- **Proof path:** `CPpa` owns `siteWorkspace` and attaches it in `initStoreage` (`testEp` also covers bulk dual-write).
- Completed in [CIP-0034](../cip0034.md): accessors are base-only; storage is `ownedSites` or an attached inference workspace (IVM / PPA / SparsePpa).

## Thin façade

`gpc.GPClassification(X, Y, kernel=..., inference='ep'|'laplace')` — FTC only; EP predictive via `epLatentMeanVar` / `predict_proba`.
