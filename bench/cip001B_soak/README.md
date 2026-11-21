# CIP-001B stress / mpmath soak (2026-11-21)

Close-gate evidence for [cip001B.md](../../cip/cip001B.md) backlog `2026-11-20_cip001B-stress-mpmath-soak`.

## Commands

```bash
make USE_MATIO=1 CHECKS=full diagLqVsStable testLQAdjoint
./diagLqVsStable   # writes state dumps under this directory
./testLQAdjoint
python3 scripts/sparse_mp_reference.py \
  bench/cip001B_soak/largeN_clust_kufkfu_dtcvar_stable_state.txt \
  --approx dtcvar --tol 1e-4 --rtol 1e-5
python3 scripts/sparse_mp_reference.py \
  bench/cip001B_soak/largeN_clust_kufkfu_dtcvar_lq_state.txt \
  --approx dtcvar --tol 1e-4 --rtol 1e-5
```

## Results (summary)

### Crowded \(Z\) — \(K_{mm}\)-limited (not Seeger/C-side)

From [2026-11-21_diagLqVsStable.txt](./2026-11-21_diagLqVsStable.txt):

| Cell | stable↔lq \(\lvert\Delta\mathrm{LL}\rvert\) | FD util |
|------|---------------------------------------------|---------|
| `crowdZ_fitc` | 0 | equal (\(\sim 2\times 10^{-7}\)) |
| `crowdZ_dtcvar` | 0 | equal (\(\sim 1.60\)) — both bad |
| `crowdZ_dtc` | 0 | equal (\(\sim 1.50\)) — both bad |

LQ matches StableFactor exactly. Collapsed inducing points make \(K_{uu}\) the bottleneck; \(C\approx I\) so the LQ of \([I,\sqrt{\beta}B]\) cannot help. Documented limit, not a regression.

### Large-\(N\) clustered — \(K_{uf}K_{fu}\)-limited (Seeger/C-side)

Head-to-head GP LL/grad (`largeN_clust_kufkfu_dtcvar`, \(N=2000\), \(M=25\), \(\beta=10^6\)):

- \(\lvert\Delta\mathrm{LL}\rvert(\mathrm{stable}\to\mathrm{lq})\approx 9.3\times 10^{-10}\)
- FD relative error \(\sim 0.298\) for all three modes; LQ marginally tighter than StableFactor

Logdet geometry gate ([2026-11-21_testLQAdjoint.txt](./2026-11-21_testLQAdjoint.txt)):

- \(\kappa(K_{uu})\approx 25\), \(\kappa(K_{uf}K_{fu})\approx 5\times 10^{16}\)
- \(\lvert\mathrm{ld_{chol}}-\mathrm{ld_{svd}}\rvert / \lvert\mathrm{ld_{lq}}-\mathrm{ld_{svd}}\rvert \approx 4556\)

mpmath whitened DTCVAR ([2026-11-21_mpmath_largeN.txt](./2026-11-21_mpmath_largeN.txt)), dps 60, compound white included:

| Dump | \(\lvert\Delta\mathrm{LL}\rvert\) | rel |
|------|-----------------------------------|-----|
| StableFactor | \(\approx 9.83\) | \(\approx 8.7\times 10^{-6}\) PASS (`rtol=1e-5`) |
| LQ | \(\approx 9.83\) | \(\approx 8.7\times 10^{-6}\) PASS (`rtol=1e-5`) |

Absolute double–mpmath gap of order 10 is expected on this \(\kappa\) cell; relative agreement and dual-path \(\lvert\Delta\mathrm{LL}\rvert\sim 10^{-9}\) are the soak gates.
