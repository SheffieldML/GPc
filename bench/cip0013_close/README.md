# CIP-0013 close re-benchmark (2026-11-21) — withdrawn

**Withdrawn 2026-11-22.** See [CoE](../../coe/2026-11-22_cip0013-bench-cache-reuse.md).

The CSV below was produced by a harness that did **not** perturb parameters between timed repeats, so GPc reused factorisation caches after warm-up. That violated the CIP-0013 protocol. Do not use these numbers for REQ-0007.

Historical raw (invalid for close): [2026-11-21_q2_rebench.csv](./2026-11-21_q2_rebench.csv).

Re-publish here only after a cache-busting rebench (`setOptParams` + perturbed params in the timed loop) on the reference Mac.
