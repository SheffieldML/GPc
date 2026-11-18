---
id: "0010"
title: "Documented concurrency contract for models, RNG, and BLAS nesting"
status: "Proposed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets: ["assert-and-verify", "flexibility-with-performance", "ecosystem-interoperability"]
stakeholders: ["maintainers", "python-facade-users", "multi-process-trainers"]
tags: ["concurrency", "threading", "documentation", "rng", "blas", "python"]
---

# REQ-0010: Documented concurrency contract for models, RNG, and BLAS nesting

## Description

Users and bindings can rely on a published concurrency contract: one live model graph (GP / GPLVM and its owned kernel and noise scratch) is single-threaded; independent instances with separate kernels may run on different threads once process-global mutable state is named and made safe or documented; nested BLAS threading is an explicit policy, not an accident.

**Why this matters**: [CIP-0009](../cip/cip0009.md) already made working-state caches explicit and recorded “one model, one thread.” That is the right performance trade-off (mutable caches, no mutex tax). The gap is the next rung: a process-level contract that covers `ndlutil` RNG globals, BLAS nesting when the application is already multi-threaded, and the Python façade’s GIL assumptions — so multi-model parallel training and `gpc` users do not discover races by accident.

**Who benefits**: Maintainers writing parallel experiment drivers; CIP-001D / REQ-000E Python users; anyone linking OpenBLAS in a multi-threaded host.

## Acceptance Criteria

- [ ] A short concurrency contract is published where builders and Python users will see it (at least `docs/numerics.md`, with pointers from README and/or `python/README.md`)
- [ ] The contract states: one model instance (and shared kernel scratch) → one thread; independent models with separate kernels → OK under the documented process rules
- [ ] Process-global mutable state that affects multi-instance use is named (at least `ndlutil` Mersenne Twister); either isolated (`thread_local` or an explicit RNG object) or documented as process-global with a “do not race” rule
- [ ] BLAS nesting policy is stated (application threads ⇒ set OpenBLAS / vendor BLAS to one thread, matching the existing bench practice)
- [ ] Python façade documents that one `GPRegression` (or wrapped model) is not free-threaded; GIL stay-held during `optimize` unless a later CIP proves release is safe
- [ ] Same-instance concurrent likelihood/gradient and mutexed model APIs are explicitly out of scope for this requirement

## Notes (Optional)

- Performance parallelism inside one evaluation (multi-threaded BLAS or OpenMP over kernel loops) remains [CIP-0013](../cip/cip0013.md) P7 — orthogonal to this safety/documentation requirement.
- Debug-only owner-thread asserts are allowed as HOW; they are not required for Validated if documentation + RNG policy alone meet the criteria.

## References

- **Related Tenets**: assert-and-verify, flexibility-with-performance, ecosystem-interoperability
- **Related**: CIP-0009 L1 note in [docs/numerics.md](../docs/numerics.md); [CIP-001D](../cip/cip001D.md) / [CIP-001E](../cip/cip001E.md) GIL notes; OpenBLAS multi-threaded application guidance

## Progress Updates

### 2026-11-18
Requirement created with CIP-001F (Proposed) after review of L2/L3 concurrency ladder.
