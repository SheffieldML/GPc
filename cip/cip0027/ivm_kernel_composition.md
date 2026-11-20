# CIP-0027 Phase B′ — IVM kernel composition results

Date: 2026-11-20. Diagnostic: `./diagIvmSoftener` (`tools/diagIvmSoftener.cpp`).  
Data: `examples/unitsquaregp.svml` (N=500). Active set M=50, ENTROPY, seed 42.  
Protocol: `optimise(3, 40, 15)` (re-selects each outer iter).

## Matrix

| Config | Train err | `white.variance` | `bias.variance` | predictive LL | Jaccard vs (1) |
|--------|-----------|------------------|-----------------|---------------|----------------|
| (1) RBF + Bias + White | 0.436 | ~1e-6 | ~0.001 | −345.90 | 1.00 |
| (2) RBF only | 0.436 | — | — | −342.47 | **0.32** |
| (3) RBF + White | 0.436 | ~1e-6 | — | −345.90 | **0.92** |

Train error stayed high on this short budget (classification is hard with M=50 and few outer iters). The CIP decision metric is **active-set overlap / softener behaviour**, not a polished accuracy bake-off.

## Decisions (locked for CIP-0027)

1. **Keep auto White on IVM.** RBF-only diverges sharply in active-set selection (Jaccard 0.32). Softness that is trained still belongs in White-in-\(K\) while probit \(\sigma^2\) stays fixed at \(10^{-6}\). Documented exception vs regression/PPA/Python defaults.
2. **Drop auto Bias on IVM.** RBF+White matches status quo LL and nearly matches the active set (Jaccard 0.92); kernel Bias is redundant with probit bias. Explicit `-k bias` remains available.
3. **Trainable probit \(\sigma^2\) spike:** not run in this pass; still deferred to the follow-on likelihood / inference CIP (`ivm_softener.md` Option B).

## Longer-budget confirmation (2026-11-20)

Protocol: M=80, `optimise(4, 50, 15)`, same seed. Gated in `./testIvmSoftener`.

| Config | Jaccard vs (1) | |LL − LL₍₁₎| |
|--------|----------------|---------------|
| RBF only | ≈ 0.09 | ≈ 3.7 |
| RBF + White | 1.00 | ≈ 5×10⁻⁴ |

Selection divergence sharpens with budget; RBF+White remains interchangeable with status quo. Crescent labelled (N=37) is too easy for softener discrimination (all configs perfect) — unitsquare stays the gate.

Additional unit gates in `testIvmSoftener`: `getWhite()` matches `white.variance`; probit σ² fixed at 10⁻⁶ after optimise; White changes `selectPoints` vs RBF-only; stream round-trip preserves active set (full `equals` still blocked by stream omitting `inactiveSet`).

## Reproduce

```sh
make diagIvmSoftener && ./diagIvmSoftener
make testIvmSoftener && ./testIvmSoftener
```

