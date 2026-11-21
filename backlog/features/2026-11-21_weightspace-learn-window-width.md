---
id: "2026-11-21_weightspace-learn-window-width"
title: "Wire global learnable window width s (learnS + wsGradS)"
status: "In Progress"
priority: "High"
created: "2026-11-21"
last_updated: "2026-11-21"
category: "features"
related_cips: ["0028", "001C"]
owner: "Neil Lawrence"
dependencies: ["2026-11-21_weightspace-inducing-fixed-default"]
tags:
- backlog
- weightspace
- numerics
- inducing
- gradients
---

# Task: Global learnable window width \(s\) for weight-space inducing

## Description

CIP-001C / 0028 ship \(s\) (`window_width`) **fixed by default** for conditioning. That is intentional policy, not a product bug. However `CInducingWeightSpace` already has a `learnS` flag and param pack/unpack, while `gradFromKuuKuf` does `(void)learnS` — no \(\partial L/\partial s\) is returned (kernel-level `wsGradS` / `wsRidgeGradS` already exist and are FD-gated in `testCGp`). Softness of the smoothed erf units is therefore stuck unless the user hand-sweeps \(s\).

This backlog wires **one global** learnable \(s\) end-to-end (C++ grads + CGp packing + Python opt-in). Per-inducing \(s_m\) stays out of scope (future CIP).

## Acceptance Criteria

- [ ] With `learnS=true`, `gradFromKuuKuf` accumulates \(\partial L/\partial s\) from \(K_{uu}\) and \(K_{uf}\) (ridge and isotropic paths)
- [ ] `CGp` opt-param packing includes \(s\) when inducing params are free and `learnS` is on; analytic grad matches FD
- [ ] Python can opt in (e.g. `learn_window_width=True`) with `inducing_fixed=False`; document conditioning caution
- [ ] Per-unit \(s_m\) explicitly deferred (not implemented)

## Implementation Notes

- Kernel hooks: `wsRidgeGradS` / `wsGradS` for \(K_{fu}\); add \(\langle gK_{uu},\partial K_{uu}/\partial s\rangle\) (analytic or tight FD on `fillKuu*`).
- `CSvgp::packInducingGrad` currently leaves learnS slots at 0 — align or document SVGP out of scope for this backlog.
- Keep default `learnS=false`; demos may enable it after free centres land.

## Related

- CIP: 0028, 001C
- Sibling: `2026-11-21_weightspace-inducing-fixed-default`
- Non-goal: per-inducing \(s_m\) (new CIP if pursued)

## Progress Updates

### 2026-11-21

Task created; `learnS` packing exists, gradient path voided.
