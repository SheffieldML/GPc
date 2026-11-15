---
id: "2026-11-15_gnuplot-term-and-style"
title: "Fix gnuplot terminal selection and modernise plot style"
status: "Completed"
priority: "Low"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "features"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- features
- gnuplot
- visualisation
- macos
- gpy
---

# Task: Gnuplot terminal and modern style

## Description

Generated `*_plot.gp` scripts hard-coded `set term x11` and used the removed `table` terminal for contour extraction. On current macOS Homebrew builds, X11 is unavailable and `GNUTERM=x11` is often still exported by shells. Plots also lacked GPy’s Tango colour language and filled confidence bands.

## Acceptance Criteria

- [x] Generated scripts pick a usable interactive terminal (`qt` default; `GPC_GNUTERM` override) instead of hard-coded `x11`
- [x] File output path uses `pngcairo` via `GPLOT_PNG='name.png'`
- [x] README notes match current Homebrew gnuplot install (drop obsolete `--with-x`)
- [x] 1-D GP mean + CI use GPy-like colours with filled ±2σ bands; data black; inducing dark red
- [x] Classification / IVM / PPA / GPLVM scripts share the same palette
- [x] `./gp gnuplot … && gnuplot -e "GPLOT_PNG=…" *_plot.gp` works on stock Homebrew gnuplot (no XQuartz)
- [x] Contour extraction uses `set table` / `unset table`; GPLVM plot data is unheaded for gnuplot

## Implementation Notes

- Shared helpers on `CClctrl`: `writeGnuplotStyleHeader`, `writeGnuplotTermSelect`, `writeGnuplotFinish`
- Do not honour `GNUTERM` (often still `x11`); use `GPC_GNUTERM` for interactive overrides
- Regenerated README PNGs: sinc, unitsquaregp, ard_gaussian_clusters, semisupercrescent (+ labels_only), oil100, oil1000
- Swagger examples skipped (no `examples/swagger.svml` in tree)

## Related

- Code: `CClctrl.{h,cpp}`, `gp.cpp`, `ivm.cpp`, `ppa.cpp`, `gplvm.cpp`
- Docs: `README.md`
- Style reference: GPy `GPy/plotting/Tango.py`, `GPy/plotting/matplot_dep/defaults.py`

## Progress Updates

### 2026-11-15

Task created with Proposed status after CIP-0012 land.

### 2026-11-15 (implementation)

Completed on `gnuplot-term-and-style`: helpers, `set table`, Tango styles, README updates, regenerated PNGs.
