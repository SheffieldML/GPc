# CIP-0033 Phase 0 — `CNoise` site / data inventory (2026-11-21)

## Accept locks

- **FTC-first EP** on `CGp` (sparse EP / EPDTC deferred).
- **GPy mirror** for sites and Bernoulli–probit moments ([gpy_mirror.md](./gpy_mirror.md)).
- IVM White softener **unchanged** in this CIP.

## Where sites / targets live today

| Location | What | Migration target |
|---|---|---|
| `CNoise::py` | pointer to targets | model / dataset owner; likelihood reads batches |
| `CNoise::mu`, `varSigma` | ADF / predictive caches written by IVM, PPA, `CGp` | inference workspaces (`CEpSites`, IVM/PPA site buffers) |
| `CIvm` | writes noise mus / varSigmas during selection | keep behaviour; back by workspace later |
| `CPpa` | `noise.setMus(fBar)` / `setVarSigmas` | PPA site state → PPA workspace |
| `CGp` | Gaussian path only for hypers; noise for predict/`out` | EP via `CEpInference` then wire objective |
| `ndlnoise::*` | Gaussian batched helpers (CIP-0027) | add Bernoulli–probit `ep_moments` |

## Hot-path touch counts (approx, 2026-11-21)

- `CNoise.cpp`: heavy `getMu` / `getVarSigma` / `getTarget` use (Gaussian, probit, ordered, ncnm).
- `CIvm.cpp` / `CPpa.cpp`: setMus / setVarSigmas after latent updates.
- Python façade: no EP yet; classification checklist still IVM substitute.

## Phase 0 exit

- [x] GPy mirror note
- [x] This inventory
- [x] `ep_moments` signature in `CNoise.h` / `ndlnoise`
- [ ] Full dual-write migration of IVM/PPA (Phase A+)
